"""Explanations must add up exactly, map back to real input features, and recover a
signal deliberately planted in synthetic data."""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from readmission.explainability import (
    FEATURE_LABELS,
    contribution_importance,
    explained_pipeline,
    feature_contributions,
    feature_importance_plot,
    permutation_importance_table,
    source_feature,
    top_factors,
)
from readmission.preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessing_pipeline,
    engineer_features,
    make_binary_target,
)
from tests.conftest import make_synthetic_cohort


@pytest.fixture(scope="module")
def cohort():
    raw = make_synthetic_cohort()
    return engineer_features(raw), make_binary_target(raw["readmitted"])


def fit_pipeline(model, X, y) -> Pipeline:
    pipeline = Pipeline(
        steps=[("preprocessing", build_preprocessing_pipeline()), ("model", model)]
    )
    return pipeline.fit(X, y)


@pytest.fixture(scope="module")
def xgb_pipeline(cohort):
    X, y = cohort
    model = XGBClassifier(
        n_estimators=40, max_depth=3, learning_rate=0.3, random_state=0, n_jobs=1
    )
    return fit_pipeline(model, X, y)


@pytest.fixture(scope="module")
def logistic_pipeline(cohort):
    X, y = cohort
    return fit_pipeline(LogisticRegression(max_iter=1000), X, y)


# ---------------------------------------------------------------------------
# Mapping preprocessed columns back to input features
# ---------------------------------------------------------------------------


class TestSourceFeature:
    @pytest.mark.parametrize(
        ("output_name", "expected"),
        [
            ("number_inpatient", "number_inpatient"),
            ("gender_Male", "gender"),
            ("admission_type_emergency", "admission_type"),
            ("diag_1_category_circulatory", "diag_1_category"),
        ],
    )
    def test_known_names(self, output_name, expected):
        assert source_feature(output_name) == expected

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            source_feature("not_a_feature")

    def test_every_preprocessed_column_maps_to_an_input_feature(self, xgb_pipeline):
        names = xgb_pipeline.named_steps["preprocessing"].get_feature_names_out()
        assert {source_feature(n) for n in names} == set(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Contributions
# ---------------------------------------------------------------------------


class TestFeatureContributions:
    def test_xgboost_contributions_add_up_to_log_odds(self, cohort, xgb_pipeline):
        """SHAP's defining property: base value + contributions == model output."""
        X, _ = cohort
        contributions, base = feature_contributions(xgb_pipeline, X)
        Xt = xgb_pipeline.named_steps["preprocessing"].transform(X)
        margin = xgb_pipeline.named_steps["model"].predict(Xt, output_margin=True)
        np.testing.assert_allclose(base + contributions.sum(axis=1), margin, atol=1e-4)

    def test_logistic_contributions_add_up_to_log_odds(self, cohort, logistic_pipeline):
        X, _ = cohort
        contributions, base = feature_contributions(logistic_pipeline, X)
        Xt = logistic_pipeline.named_steps["preprocessing"].transform(X)
        log_odds = logistic_pipeline.named_steps["model"].decision_function(Xt)
        np.testing.assert_allclose(
            base + contributions.sum(axis=1), log_odds, atol=1e-8
        )

    @pytest.mark.parametrize("pipeline_name", ["xgb_pipeline", "logistic_pipeline"])
    def test_one_column_per_input_feature(self, cohort, pipeline_name, request):
        X, _ = cohort
        pipeline = request.getfixturevalue(pipeline_name)
        contributions, _ = feature_contributions(pipeline, X)
        assert list(contributions.columns) == FEATURE_COLUMNS
        assert len(contributions) == len(X)

    def test_unsupported_model_raises(self, cohort):
        X, y = cohort
        pipeline = fit_pipeline(DummyClassifier(), X, y)
        with pytest.raises(TypeError):
            feature_contributions(pipeline, X)


# ---------------------------------------------------------------------------
# Importance rankings
# ---------------------------------------------------------------------------


class TestImportance:
    def test_planted_signal_has_largest_contribution(self, cohort, xgb_pipeline):
        X, _ = cohort
        contributions, _ = feature_contributions(xgb_pipeline, X)
        rows = contribution_importance(contributions, X)
        assert rows[0]["feature"] == "number_inpatient"
        # more prior visits -> higher predicted risk
        assert rows[0]["value_vs_contribution_corr"] > 0.5

    def test_categorical_rows_name_their_extreme_levels(self, cohort, xgb_pipeline):
        X, _ = cohort
        contributions, _ = feature_contributions(xgb_pipeline, X)
        discharge = next(
            r
            for r in contribution_importance(contributions, X)
            if r["feature"] == "discharge_disposition"
        )
        assert discharge["highest_risk_level"] in set(X["discharge_disposition"])
        assert discharge["lowest_risk_level"] in set(X["discharge_disposition"])

    def test_rows_are_sorted_by_importance(self, cohort, xgb_pipeline):
        X, _ = cohort
        contributions, _ = feature_contributions(xgb_pipeline, X)
        values = [
            r["mean_abs_contribution"]
            for r in contribution_importance(contributions, X)
        ]
        assert values == sorted(values, reverse=True)

    def test_permutation_importance_finds_planted_signal(self, cohort, xgb_pipeline):
        X, y = cohort
        rows = permutation_importance_table(xgb_pipeline, X, y, n_repeats=3)
        assert rows[0]["feature"] == "number_inpatient"
        assert rows[0]["roc_auc_drop"] > 0.1
        assert {r["feature"] for r in rows} == set(FEATURE_COLUMNS)


def test_plot_is_written(cohort, xgb_pipeline, tmp_path):
    X, y = cohort
    contributions, _ = feature_contributions(xgb_pipeline, X)
    explainability = {
        "contribution_importance": contribution_importance(contributions, X),
        "permutation_importance": permutation_importance_table(
            xgb_pipeline, X, y, n_repeats=2
        ),
    }
    path = tmp_path / "importance.png"
    feature_importance_plot(explainability, path)
    assert path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Per-patient explanations (served by the API)
# ---------------------------------------------------------------------------


def calibrate(pipeline, X, y):
    """Wrap a fitted pipeline the same way train.py does."""
    return CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid").fit(X, y)


class TestPerPatientExplanations:
    def test_every_feature_has_a_plain_language_label(self):
        assert set(FEATURE_LABELS) == set(FEATURE_COLUMNS)

    def test_explained_pipeline_unwraps_the_calibrated_model(
        self, cohort, xgb_pipeline
    ):
        X, y = cohort
        assert explained_pipeline(calibrate(xgb_pipeline, X, y)) is xgb_pipeline

    def test_logistic_regression_is_refused(self, cohort, logistic_pipeline):
        """Its contributions are relative to the rows passed in; for one patient that
        is the patient itself, so every contribution would silently be zero."""
        X, y = cohort
        with pytest.raises(TypeError, match="XGBoost"):
            explained_pipeline(calibrate(logistic_pipeline, X, y))

    @pytest.mark.parametrize(
        ("inpatient", "effect"), [(5, "raises risk"), (0, "lowers risk")]
    )
    def test_planted_signal_is_the_top_factor(
        self, cohort, xgb_pipeline, inpatient, effect
    ):
        X, _ = cohort
        patient = X[X["number_inpatient"] == inpatient].iloc[[0]]
        factors = top_factors(xgb_pipeline, patient)
        assert factors[0]["feature"] == "number_inpatient"
        assert factors[0]["effect"] == effect
        assert factors[0]["value"] == inpatient

    def test_factors_are_ranked_and_signed_consistently(self, cohort, xgb_pipeline):
        X, _ = cohort
        factors = top_factors(xgb_pipeline, X.iloc[[0]], n=5)
        sizes = [abs(f["log_odds_contribution"]) for f in factors]
        assert len(factors) == 5
        assert sizes == sorted(sizes, reverse=True)
        for f in factors:
            assert (f["log_odds_contribution"] > 0) == (f["effect"] == "raises risk")

    def test_explains_one_patient_at_a_time(self, cohort, xgb_pipeline):
        X, _ = cohort
        with pytest.raises(ValueError):
            top_factors(xgb_pipeline, X.iloc[:2])
