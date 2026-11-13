import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from readmission.config import (
    RISK_TIER_ORDER,
    RISK_TIER_THRESHOLDS,
    assign_risk_tier,
)
from readmission.evaluation import (
    MIN_SUBGROUP_POSITIVES,
    age_band,
    capacity_table,
    decision_curve,
    evaluate,
    expected_calibration_error,
    incumbent_baseline,
    lift_table,
    murphy_decomposition,
    no_skill_baselines,
    risk_tier_summary,
    subgroup_columns,
    subgroup_metrics,
)


@pytest.fixture
def imbalanced_labels():
    """1,000 labels at exactly 10% prevalence."""
    return np.array([1] * 100 + [0] * 900)


# ---------------------------------------------------------------------------
# No-skill baselines
# ---------------------------------------------------------------------------


class TestNoSkillBaselines:
    def test_balanced_case_has_known_values(self):
        """At p=0.5: accuracy 0.5, Brier 0.25, log loss ln(2)."""
        baselines = no_skill_baselines(np.array([0, 1]))
        assert baselines["accuracy"] == pytest.approx(0.5)
        assert baselines["brier"] == pytest.approx(0.25)
        assert baselines["log_loss"] == pytest.approx(np.log(2))

    def test_imbalanced_case(self, imbalanced_labels):
        baselines = no_skill_baselines(imbalanced_labels)
        assert baselines["prevalence"] == pytest.approx(0.10)
        assert baselines["accuracy"] == pytest.approx(0.90)  # predict majority
        assert baselines["pr_auc"] == pytest.approx(0.10)  # equals prevalence
        assert baselines["brier"] == pytest.approx(0.09)  # p(1-p)

    def test_roc_auc_floor_is_prevalence_independent(self, imbalanced_labels):
        """The reason ROC-AUC is comparable across datasets and PR-AUC is not."""
        assert no_skill_baselines(imbalanced_labels)["roc_auc"] == 0.5
        assert no_skill_baselines(np.array([0, 1]))["roc_auc"] == 0.5

    def test_brier_formula_matches_empirical_constant_forecast(self, imbalanced_labels):
        baselines = no_skill_baselines(imbalanced_labels)
        constant = np.full(len(imbalanced_labels), baselines["prevalence"])
        assert baselines["brier"] == pytest.approx(
            brier_score_loss(imbalanced_labels, constant)
        )


# ---------------------------------------------------------------------------
# Lift
# ---------------------------------------------------------------------------


class TestLiftTable:
    def test_perfect_ranker_achieves_maximum_lift(self, imbalanced_labels):
        """All positives ranked first -> precision 1.0 in the top 10%."""
        perfect = imbalanced_labels.astype(float)
        row = lift_table(imbalanced_labels, perfect, fractions=(0.10,))[0]
        assert row["precision"] == pytest.approx(1.0)
        assert row["lift"] == pytest.approx(10.0)  # 1.0 / 0.10
        assert row["recall"] == pytest.approx(1.0)

    def test_uninformative_scores_give_lift_near_one(self, imbalanced_labels):
        rng = np.random.default_rng(0)
        rows = lift_table(imbalanced_labels, rng.random(len(imbalanced_labels)))
        assert rows[-1]["lift"] == pytest.approx(1.0, abs=0.5)

    def test_recall_increases_with_k(self, imbalanced_labels):
        rng = np.random.default_rng(0)
        recalls = [r["recall"] for r in lift_table(imbalanced_labels, rng.random(1000))]
        assert recalls == sorted(recalls)

    def test_n_flagged_matches_requested_fraction(self, imbalanced_labels):
        row = lift_table(imbalanced_labels, np.zeros(1000), fractions=(0.05,))[0]
        assert row["n_flagged"] == 50


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_constant_scores_zero(self, imbalanced_labels):
        """Predicting the base rate for everyone is perfectly calibrated —
        and completely uninformative. Calibration is not skill."""
        constant = np.full(len(imbalanced_labels), 0.10)
        assert expected_calibration_error(imbalanced_labels, constant) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_systematic_inflation_is_detected(self, imbalanced_labels):
        """A model predicting 60% on a 10% cohort is badly miscalibrated."""
        inflated = np.full(len(imbalanced_labels), 0.60)
        assert expected_calibration_error(imbalanced_labels, inflated) == pytest.approx(
            0.50, abs=1e-9
        )

    def test_monotone_rescaling_changes_ece_but_not_ranking(self, imbalanced_labels):
        rng = np.random.default_rng(1)
        proba = np.clip(rng.normal(0.10, 0.03, len(imbalanced_labels)), 0.001, 0.999)
        assert expected_calibration_error(
            imbalanced_labels, proba
        ) < expected_calibration_error(imbalanced_labels, proba / 3)


class TestMurphyDecomposition:
    def test_uncertainty_depends_only_on_prevalence(self, imbalanced_labels):
        rng = np.random.default_rng(2)
        parts = murphy_decomposition(imbalanced_labels, rng.random(1000))
        assert parts["uncertainty"] == pytest.approx(0.10 * 0.90)

    def test_components_reconstruct_the_brier_score(self):
        """reliability - resolution + uncertainty == Brier.

        The identity is exact only when forecasts are constant within each bin,
        so this uses two equally sized forecast groups that quantile-bin cleanly:
        500 patients forecast at 0.40 (150 readmitted) and 500 at 0.08 (50
        readmitted), i.e. both bins are miscalibrated and reliability > 0.
        """
        y_true = np.concatenate(
            [np.ones(150), np.zeros(350), np.ones(50), np.zeros(450)]
        )
        proba = np.concatenate([np.full(500, 0.40), np.full(500, 0.08)])

        parts = murphy_decomposition(y_true, proba, n_bins=2)
        reconstructed = (
            parts["reliability"] - parts["resolution"] + parts["uncertainty"]
        )
        assert reconstructed == pytest.approx(brier_score_loss(y_true, proba), abs=1e-9)
        assert parts["reliability"] > 0

    def test_well_calibrated_model_has_near_zero_reliability(self, imbalanced_labels):
        constant = np.full(len(imbalanced_labels), 0.10)
        parts = murphy_decomposition(imbalanced_labels, constant)
        assert parts["reliability"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------


class TestAssignRiskTier:
    """Tier boundaries are a product contract — they decide who gets an
    intervention — so the edges are pinned explicitly."""

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.0, "low"),
            (0.119, "low"),
            (0.12, "moderate"),  # threshold is inclusive
            (0.159, "moderate"),
            (0.16, "high"),  # threshold is inclusive
            (1.0, "high"),
        ],
    )
    def test_boundaries(self, probability, expected):
        assert assign_risk_tier(probability) == expected

    def test_every_probability_maps_to_a_known_tier(self):
        for probability in np.linspace(0, 1, 101):
            assert assign_risk_tier(float(probability)) in RISK_TIER_ORDER

    def test_lowest_tier_has_no_threshold_entry(self):
        """It is the catch-all; a threshold for it could never bind."""
        assert RISK_TIER_ORDER[0] not in RISK_TIER_THRESHOLDS

    def test_thresholds_only_name_known_tiers(self):
        assert set(RISK_TIER_THRESHOLDS) <= set(RISK_TIER_ORDER)

    def test_order_runs_from_lowest_to_highest_risk(self):
        thresholds = [RISK_TIER_THRESHOLDS.get(t, 0.0) for t in RISK_TIER_ORDER]
        assert thresholds == sorted(thresholds)


class TestRiskTierSummary:
    def test_tiers_partition_the_cohort(self, imbalanced_labels):
        rng = np.random.default_rng(3)
        rows = risk_tier_summary(imbalanced_labels, rng.random(1000))
        assert sum(r["n_patients"] for r in rows) == 1000
        assert sum(r["share_of_cohort"] for r in rows) == pytest.approx(1.0)

    def test_empty_tier_reports_none_not_nan(self, imbalanced_labels):
        """Every patient below 10% -> moderate and high tiers are empty."""
        rows = {
            r["tier"]: r
            for r in risk_tier_summary(imbalanced_labels, np.full(1000, 0.01))
        }
        assert rows["low"]["n_patients"] == 1000
        assert rows["high"]["observed_readmission_rate"] is None

    def test_tier_assignment_respects_thresholds(self, imbalanced_labels):
        proba = np.concatenate(
            [np.full(500, 0.05), np.full(300, 0.15), np.full(200, 0.25)]
        )
        rows = {
            r["tier"]: r["n_patients"]
            for r in risk_tier_summary(imbalanced_labels, proba)
        }
        assert rows == {"low": 500, "moderate": 300, "high": 200}


# ---------------------------------------------------------------------------
# Subgroups
# ---------------------------------------------------------------------------


class TestCapacityTable:
    def test_volumes_and_rates_for_a_known_split(self, imbalanced_labels):
        """Score the 100 readmissions plus 100 non-readmissions at 0.30, the rest at
        0.05. A 0.20 cutoff flags 200 of 1,000 patients, half of them readmitted."""
        proba = np.concatenate([np.full(200, 0.30), np.full(800, 0.05)])
        row = capacity_table(imbalanced_labels, proba, cutoffs=(0.20,))[0]
        assert row["flagged_per_100"] == pytest.approx(20.0)
        assert row["precision"] == pytest.approx(0.5)
        assert row["recall"] == pytest.approx(1.0)
        assert row["readmissions_flagged_per_100"] == pytest.approx(10.0)
        assert row["number_needed_to_flag"] == pytest.approx(2.0)

    def test_cutoff_above_every_score_flags_no_one(self, imbalanced_labels):
        row = capacity_table(imbalanced_labels, np.full(1000, 0.05), cutoffs=(0.5,))[0]
        assert row["flagged_per_100"] == 0
        assert row["precision"] is None
        assert row["number_needed_to_flag"] is None

    def test_raising_the_cutoff_never_flags_more_patients(self, imbalanced_labels):
        rng = np.random.default_rng(8)
        rows = capacity_table(imbalanced_labels, rng.random(1000), (0.1, 0.3, 0.5, 0.9))
        volumes = [r["flagged_per_100"] for r in rows]
        assert volumes == sorted(volumes, reverse=True)


class TestDecisionCurve:
    def test_hand_computed_net_benefit(self, imbalanced_labels):
        """Same split as the capacity test: at t = 0.20, 100 true and 100 false
        positives out of 1,000 -> 0.10 - 0.10 * (0.20 / 0.80) = 0.075."""
        proba = np.concatenate([np.full(200, 0.30), np.full(800, 0.05)])
        row = decision_curve(imbalanced_labels, proba, thresholds=(0.20,))[0]
        assert row["net_benefit_model"] == pytest.approx(0.075)

    def test_treat_all_equals_prevalence_at_zero_threshold(self, imbalanced_labels):
        row = decision_curve(imbalanced_labels, np.zeros(1000), thresholds=(0.0,))[0]
        assert row["net_benefit_treat_all"] == pytest.approx(0.10)

    def test_perfect_model_captures_all_benefit(self, imbalanced_labels):
        """Flagging exactly the readmitted patients has no false-alarm cost."""
        perfect = imbalanced_labels.astype(float)
        rows = decision_curve(imbalanced_labels, perfect, thresholds=(0.1, 0.3, 0.5))
        assert all(r["net_benefit_model"] == pytest.approx(0.10) for r in rows)

    def test_flagging_no_one_has_zero_net_benefit(self, imbalanced_labels):
        row = decision_curve(imbalanced_labels, np.zeros(1000), thresholds=(0.5,))[0]
        assert row["net_benefit_model"] == 0


class TestAgeBand:
    @pytest.mark.parametrize(
        ("midpoint", "expected"),
        [
            (5, "<50"),
            (45, "<50"),
            (55, "50-69"),
            (65, "50-69"),
            (75, "70-79"),
            (95, "80+"),
        ],
    )
    def test_dataset_midpoints_map_to_bands(self, midpoint, expected):
        assert age_band(pd.Series([midpoint])).iloc[0] == expected


class TestSubgroupColumns:
    def test_uses_engineered_feature_columns(self, raw_df):
        from readmission.preprocessing import clean_encounters, engineer_features

        X = engineer_features(clean_encounters(raw_df))
        columns = subgroup_columns(X)
        assert set(columns) == {"race", "gender", "age_band"}
        assert all(len(col) == len(X) for col in columns.values())


class TestSubgroupMetrics:
    @staticmethod
    def two_groups(proba_a, proba_b, y_a, y_b):
        """Stack two groups' scores and labels into cohort-level arrays."""
        return (
            np.concatenate([y_a, y_b]),
            np.concatenate([proba_a, proba_b]),
            np.array(["A"] * len(y_a) + ["B"] * len(y_b)),
        )

    def test_groups_partition_the_cohort(self, imbalanced_labels):
        rng = np.random.default_rng(6)
        groups = rng.choice(["x", "y", "z"], size=1000)
        rows = subgroup_metrics(imbalanced_labels, rng.random(1000), groups, n_boot=20)
        assert sum(r["n_patients"] for r in rows) == 1000
        assert sum(r["n_readmitted"] for r in rows) == 100

    def test_largest_group_listed_first(self):
        y = np.array([0, 1] * 300)
        groups = np.array(["small"] * 100 + ["large"] * 500)
        rows = subgroup_metrics(y, np.linspace(0, 1, 600), groups, n_boot=20)
        assert [r["group"] for r in rows] == ["large", "small"]

    def test_detects_group_specific_overprediction(self):
        """Both groups readmit at 10%, but group B is predicted at 30%.
        A cohort-level calibration check would blur this; per-group gaps expose it."""
        y_group = np.array([1] * 50 + [0] * 450)
        y, proba, groups = self.two_groups(
            np.full(500, 0.10), np.full(500, 0.30), y_group, y_group
        )
        rows = {r["group"]: r for r in subgroup_metrics(y, proba, groups, n_boot=20)}
        assert rows["A"]["calibration_gap"] == pytest.approx(0.0)
        assert rows["B"]["calibration_gap"] == pytest.approx(0.20)

    def test_detects_unequal_recall_at_shared_cutoff(self):
        """Group A's readmissions score high and group B's score low, so a single
        cutoff catches every A readmission and no B readmission."""
        y_group = np.array([1] * 50 + [0] * 450)
        proba_a = np.concatenate([np.full(50, 0.9), np.full(450, 0.05)])
        proba_b = np.concatenate([np.full(50, 0.06), np.full(450, 0.05)])
        y, proba, groups = self.two_groups(proba_a, proba_b, y_group, y_group)
        rows = {
            r["group"]: r
            for r in subgroup_metrics(y, proba, groups, flag_threshold=0.5, n_boot=20)
        }
        assert rows["A"]["recall_at_flag"] == pytest.approx(1.0)
        assert rows["B"]["recall_at_flag"] == pytest.approx(0.0)

    def test_single_class_group_has_no_auc(self):
        """AUC is undefined without both outcomes; report None rather than crash."""
        y = np.array([0, 1] * 50 + [0] * 20)
        groups = np.array(["mixed"] * 100 + ["no_readmissions"] * 20)
        rows = {
            r["group"]: r
            for r in subgroup_metrics(y, np.linspace(0, 1, 120), groups, n_boot=20)
        }
        assert rows["no_readmissions"]["roc_auc"] is None
        assert rows["no_readmissions"]["roc_auc_ci"] == [None, None]
        assert rows["no_readmissions"]["recall_at_flag"] is None

    def test_small_groups_are_flagged_low_support(self):
        n_pos = MIN_SUBGROUP_POSITIVES - 1
        y = np.array([1] * n_pos + [0] * 200)
        rows = subgroup_metrics(
            y, np.linspace(0, 1, len(y)), np.full(len(y), "g"), n_boot=20
        )
        assert rows[0]["low_support"] is True

    def test_auc_interval_brackets_the_estimate_and_is_reproducible(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 400)
        proba = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
        groups = np.full(400, "g")
        first = subgroup_metrics(y, proba, groups, random_state=1)[0]
        again = subgroup_metrics(y, proba, groups, random_state=1)[0]
        low, high = first["roc_auc_ci"]
        assert low <= first["roc_auc"] <= high
        assert first["roc_auc_ci"] == again["roc_auc_ci"]


# ---------------------------------------------------------------------------
# Incumbent baseline + headline metrics
# ---------------------------------------------------------------------------


class TestIncumbentBaseline:
    def test_scores_a_single_column_without_a_model(self, imbalanced_labels):
        X = pd.DataFrame({"number_inpatient": imbalanced_labels * 3.0})
        result = incumbent_baseline(X, imbalanced_labels)
        assert result["feature"] == "number_inpatient"
        assert result["roc_auc"] == pytest.approx(1.0)  # perfectly separating here
        assert result["lift_at_10pct"] == pytest.approx(10.0)


class TestEvaluate:
    def test_returns_all_headline_metrics(self, imbalanced_labels):
        rng = np.random.default_rng(4)
        scores = evaluate(imbalanced_labels, rng.random(1000))
        assert set(scores) == {"roc_auc", "pr_auc", "brier", "log_loss"}
        assert all(isinstance(v, float) for v in scores.values())

    def test_roc_auc_is_blind_to_monotone_rescaling(self, imbalanced_labels):
        """The property that makes a separate calibration metric necessary."""
        rng = np.random.default_rng(5)
        proba = np.clip(rng.random(1000), 0.001, 0.999)
        assert evaluate(imbalanced_labels, proba)["roc_auc"] == pytest.approx(
            evaluate(imbalanced_labels, proba / 3)["roc_auc"]
        )
