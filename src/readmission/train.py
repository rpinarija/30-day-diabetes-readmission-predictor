""" Train the model for readmission prediction.

Design:
1. 80/20 stratidied train/test split
2. Both logistic regression and XGBoost models are compared with 5-fold stratified cross-validation on the
   training set. The preprocessign ColumnTransformer lives inside the pipeline, so that the imputation/scaling/encoding
   is applied to each fold of the cross-validation, ensuring there is no leakage.
3. The winning model family is then tuned with a randomized hyperparameter search (also 5-fold CV, train set only).
4. The tuned model is refit on the training set, then calibrated with isotonic regression on the test set.
5. Every headline score is reported next to two reference points: the no-skill floor and a no-ML incumbent rule.
6. The calibrated pipeline and metadata is saved to models/model.joblib, all metrics to models/metrics.json, the
   reliability diagram to models/reliability.png, and every tuning trial to models/tuning_runs.jsonl.
"""

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from readmission import __version__
from readmission.config import (
    CALIBRATION_PLOT_PATH,
    CALIBRATION_SIZE,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TEST_SIZE,
    TUNING_LOG_PATH,
)
from readmission.evaluation import (
    calibration_curve_plot,
    evaluate,
    expected_calibration_error,
    format_summary,
    incumbent_baseline,
    lift_table,
    murphy_decomposition,
    no_skill_baselines,
    risk_tier_summary,
)
from readmission.preprocessing import (
    build_preprocessing_pipeline,
    clean_encounters,
    engineer_features,
    make_binary_target,
)

CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "neg_brier_score",
}

# Hyperparameter search spaces, keyed to match each candidate's "model" pipeline step.
# Bracket the hardcoded defaults in build_candidates() on both sides rather than guessing blind.
PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "model__C": loguniform(0.001, 100),
    },
    "xgboost": {
        "model__max_depth": randint(3, 8),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__n_estimators": randint(150, 600),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 50),
    },
}

def build_candidates(scale_pos_weight: float = 1.0) -> dict[str, Pipeline]:
    """Both candidate models, each wrapped with the same preprocessing pipeline."""
    logistic = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessing", build_preprocessing_pipeline()),
                ("model", logistic),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocessing", build_preprocessing_pipeline()),
                ("model", xgb),
            ]
        ),
    }

def cross_validate_candidates(
    candidates: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series
) -> dict[str, dict[str, float]]:
    """5-fold CV for each model; return mean/std per metric."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict[str, float]] = {}
    for name, pipeline in candidates.items():
        print(f"Cross-validating {name}...")
        scores = cross_validate(pipeline, X, y, cv=cv,  scoring=CV_SCORING, n_jobs=-1)
        results[name] = {}
        for metric in CV_SCORING:
            values = scores[f"test_{metric}"]
            if metric == "brier":
                values = -values
            results[name][f"{metric}_mean"] = float(np.mean(values))
            results[name][f"{metric}_std"] = float(np.std(values))
        print(
            f"  ROC-AUC: {results[name]['roc_auc_mean']:.4f} "
            f"+/- {results[name]['roc_auc_std']:.4f}, "
            f"PR-AUC: {results[name]['pr_auc_mean']:.4f} "
            f"Brier: {results[name]['brier_mean']:.4f} "
        )
    return results

def summarize_cv_results(cv_results: dict) -> list[dict]:
    """Compact per-trial summary from a RandomizedSearchCV's cv_results_."""
    return [
        {
            "params": cv_results["params"][i],
            "mean_test_roc_auc": float(cv_results["mean_test_score"][i]),
            "std_test_roc_auc": float(cv_results["std_test_score"][i]),
            "rank": int(cv_results["rank_test_score"][i]),
        }
        for i in range(len(cv_results["params"]))
    ]

def tune_candidate(
    name: str,
    pipeline: Pipeline,
    param_distributions: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 30,
) -> tuple[Pipeline, dict]:
    """Randomized search over one candidate's hyperparameters, scored by CV ROC-AUC.

    Operates only on the training split passed in - never sees the test set.
    Returns the best estimator and a summary of the tuning results.
    """
    print(f"Tuning {name} ({n_iter} candidates, 5-fold CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    print(f"  Best ROC-AUC: {search.best_score_:.4f} with {search.best_params_}")
    summary = {
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "n_iter": n_iter,
        "trials": summarize_cv_results(search.cv_results_),
    }
    return search.best_estimator_, summary

def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} not found. Run `python scripts/download_data.py` first."
        )

    print("Loading and preparing data ...")
    raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    cleaned = clean_encounters(raw)
    X = engineer_features(cleaned)
    y = make_binary_target(cleaned["readmitted"])
    print(f"  {len(raw):,} raw encounters -> {len(X):,} after encounters")
    print(f"  possitive class (readmitted <30d): {y.mean():.3%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    candidates = build_candidates(scale_pos_weight)
    cv_results = cross_validate_candidates(candidates, X_train, y_train)

    best_model_name = max(cv_results, key=lambda name: cv_results[name]["roc_auc_mean"])
    print(f"Selected model family: {best_model_name}")

    tuned_estimator, tuning_summary = tune_candidate(
        best_model_name,
        candidates[best_model_name],
        PARAM_DISTRIBUTIONS[best_model_name],
        X_train,
        y_train,
    )

    # Refit winning model on a fit split, calibrate on a disjoint calibration split
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train,
        y_train,
        test_size=CALIBRATION_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE
    )
    # tuned_estimator was refit by RandomizedSearchCV on all of X_train/y_train; clone
    # it back to an unfit pipeline so the fit/calibration split stays disjoint, same as
    # the untuned flow this replaces.
    best_pipeline = clone(tuned_estimator)
    best_pipeline.fit(X_fit, y_fit)
    uncalibrated_proba = best_pipeline.predict_proba(X_test)[:, 1]

    print(f"Calibrating {best_model_name} with isotonic regression ...")
    calibrated = CalibratedClassifierCV(FrozenEstimator(best_pipeline), method="isotonic")
    calibrated.fit(X_calib, y_calib)
    calibrated_proba = calibrated.predict_proba(X_test)[:, 1]

    metrics = {
        "model_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(),
        "selected_model": best_model_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate": float(y.mean()),
        "baselines": {
            "no_skill": no_skill_baselines(y_test),
            "incumbent": incumbent_baseline(X_test, y_test),
        },
        "tuning": {
            "best_params": tuning_summary["best_params"],
            "best_cv_roc_auc": tuning_summary["best_cv_roc_auc"],
            "n_iter": tuning_summary["n_iter"],
        },
        "test_uncalibrated": evaluate(y_test, uncalibrated_proba),
        "test_calibrated": evaluate(y_test, calibrated_proba),
        "calibration": {
            "ece_uncalibrated": expected_calibration_error(y_test, uncalibrated_proba),
            "ece_calibrated": expected_calibration_error(y_test, calibrated_proba),
            "murphy_uncalibrated": murphy_decomposition(y_test, uncalibrated_proba),
            "murphy_calibrated": murphy_decomposition(y_test, calibrated_proba),
        },
        "lift_table": lift_table(y_test, calibrated_proba),
        "risk_tiers": risk_tier_summary(y_test, calibrated_proba),
    }
    print(format_summary(metrics))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": calibrated,
        "model_type": best_model_name,
        "model_version": __version__,
        "trained_at": metrics["trained_at"],
        "metrics": {"test": metrics["test_calibrated"],},
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    calibration_curve_plot(
        y_test, uncalibrated_proba, calibrated_proba, CALIBRATION_PLOT_PATH
    )
    tuning_record = {
        "trained_at": metrics["trained_at"],
        "model_version": __version__,
        "selected_model": best_model_name,
        "n_iter": tuning_summary["n_iter"],
        "trials": tuning_summary["trials"],
    }
    with TUNING_LOG_PATH.open("a") as f:
        f.write(json.dumps(tuning_record) + "\n")
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved calibration curve to {CALIBRATION_PLOT_PATH}")
    print(f"Appended tuning trials to {TUNING_LOG_PATH}")


if __name__ == "__main__":
    main()
