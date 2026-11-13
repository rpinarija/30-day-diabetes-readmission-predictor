"""Model explanations: per-patient feature contributions and global feature importance.

Two complementary views:
- Contributions (SHAP values) explain individual predictions: how much each feature pushed
  one patient's score up or down. Averaged over the cohort they show what drives the model.
- Permutation importance measures how much ranking performance (ROC-AUC) depends on each
  feature, by shuffling it and re-scoring. It is model-agnostic and uses held-out data.

Contributions explain the model's log-odds score before isotonic calibration. Calibration
is monotone, so feature rankings and directions carry over to the calibrated probabilities,
but contribution magnitudes are in log-odds, not probability points.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from readmission.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)

matplotlib.use("Agg")

# Chart tokens: one series per panel, so a single hue; text uses ink colors, not the bar color.
SURFACE = "#fcfcfb"
BAR_COLOR = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def source_feature(output_name: str) -> str:
    """Map a preprocessed column name (e.g. 'race_Caucasian') back to its input feature."""
    if output_name in FEATURE_COLUMNS:
        return output_name
    matches = [f for f in CATEGORICAL_FEATURES if output_name.startswith(f"{f}_")]
    if not matches:
        raise ValueError(f"Cannot map preprocessed column {output_name!r} to a feature")
    return max(matches, key=len)  # longest prefix wins if feature names ever overlap


def feature_contributions(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-patient contribution of each input feature to the model's log-odds score.

    - XGBoost: exact TreeSHAP values, computed natively by XGBoost.
    - Logistic regression: coefficient x (value - cohort mean), which is the exact SHAP
      value for a linear model with independent features.

    One-hot columns are summed back to the feature they came from (SHAP values are
    additive), so the columns are FEATURE_COLUMNS. For every patient,
    base_value + row sum == the model's log-odds score.
    """
    preprocessing = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    Xt = preprocessing.transform(X)

    if isinstance(model, xgb.XGBClassifier):
        contribs = model.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
        values, base_value = contribs[:, :-1], contribs[:, -1]
    elif isinstance(model, LogisticRegression):
        mean = Xt.mean(axis=0)
        values = (Xt - mean) * model.coef_[0]
        base_value = np.full(len(Xt), model.intercept_[0] + mean @ model.coef_[0])
    else:
        raise TypeError(f"No contribution method for {type(model).__name__}")

    names = preprocessing.get_feature_names_out()
    per_column = pd.DataFrame(values, columns=names, index=X.index)
    per_feature = per_column.T.groupby([source_feature(n) for n in names]).sum().T
    return per_feature[FEATURE_COLUMNS], np.asarray(base_value)


def contribution_importance(contributions: pd.DataFrame, X: pd.DataFrame) -> list[dict]:
    """Rank features by mean |contribution|, with the direction of each feature's effect.

    Numeric features report the rank correlation between the feature's value and its
    contribution (positive: higher values raise predicted risk). Categorical features
    report which level raises predicted risk the most and which lowers it the most.
    """
    rows = []
    for feature in contributions.columns:
        contribution = contributions[feature]
        row = {
            "feature": feature,
            "mean_abs_contribution": float(contribution.abs().mean()),
        }
        if feature in NUMERIC_FEATURES:
            # A feature the model never uses has zero contribution for everyone, and
            # correlation with a constant is undefined: report no direction.
            varies = X[feature].nunique() > 1 and contribution.nunique() > 1
            row["value_vs_contribution_corr"] = (
                float(X[feature].corr(contribution, method="spearman"))
                if varies
                else None
            )
        else:
            by_level = contribution.groupby(X[feature].astype(str)).mean()
            row["highest_risk_level"] = str(by_level.idxmax())
            row["lowest_risk_level"] = str(by_level.idxmin())
        rows.append(row)
    return sorted(rows, key=lambda r: r["mean_abs_contribution"], reverse=True)


def permutation_importance_table(
    model, X: pd.DataFrame, y, n_repeats: int = 5, random_state: int = 0
) -> list[dict]:
    """Drop in ROC-AUC when each feature is shuffled, on held-out data."""
    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
    )
    rows = [
        {
            "feature": feature,
            "roc_auc_drop": float(result.importances_mean[i]),
            "roc_auc_drop_std": float(result.importances_std[i]),
        }
        for i, feature in enumerate(X.columns)
    ]
    return sorted(rows, key=lambda r: r["roc_auc_drop"], reverse=True)


def format_importance(explainability: dict, top_n: int = 10) -> str:
    """Side-by-side text summary of the top features under both importance measures."""
    contribution_rows = explainability["contribution_importance"][:top_n]
    permutation_rows = explainability["permutation_importance"][:top_n]
    lines = [
        "",
        f"{'rank':>4}  {'mean |SHAP| (log-odds)':<38}{'ROC-AUC drop when shuffled':<38}",
    ]
    for rank, (c, p) in enumerate(zip(contribution_rows, permutation_rows), start=1):
        left = f"{c['feature']:<28}{c['mean_abs_contribution']:>8.4f}"
        right = f"{p['feature']:<28}{p['roc_auc_drop']:>8.4f}"
        lines.append(f"{rank:>4}  {left:<38}{right:<38}")
    return "\n".join(lines)


def feature_importance_plot(explainability: dict, path: Path, top_n: int = 15) -> None:
    """Two panels, each with its own axis: mean |SHAP| and permutation ROC-AUC drop."""
    panels = (
        (
            explainability["contribution_importance"][:top_n],
            "mean_abs_contribution",
            None,
            "What drives predictions",
            "Mean |SHAP contribution| (log-odds)",
        ),
        (
            explainability["permutation_importance"][:top_n],
            "roc_auc_drop",
            "roc_auc_drop_std",
            "What ranking accuracy depends on",
            "ROC-AUC drop when feature is shuffled",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.38 * top_n + 1.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, (rows, value_key, error_key, title, xlabel) in zip(axes, panels):
        rows = rows[::-1]  # largest at the top
        values = [r[value_key] for r in rows]
        errors = [r[error_key] for r in rows] if error_key else None
        ax.barh(
            [r["feature"] for r in rows],
            values,
            height=0.6,
            color=BAR_COLOR,
            xerr=errors,
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2},
        )
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=12, pad=10)
        ax.set_xlabel(xlabel, color=INK_SECONDARY)
        ax.tick_params(colors=INK_SECONDARY, length=0)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)
    fig.tight_layout(w_pad=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


__all__ = [
    "contribution_importance",
    "feature_contributions",
    "feature_importance_plot",
    "format_importance",
    "permutation_importance_table",
    "source_feature",
]
