"""The release gate must pass a healthy model and block each kind of regression."""

import copy
import json

import pytest

from readmission import quality_gate


def healthy_metrics() -> dict:
    """The minimal metrics structure the gate reads, with values like the current model."""
    return {
        "test_calibrated": {"roc_auc": 0.649},
        "baselines": {"incumbent": {"roc_auc": 0.552}},
        "lift_table": [
            {"top_fraction": 0.05, "lift": 2.64},
            {"top_fraction": 0.10, "lift": 2.18},
        ],
        "calibration": {"ece_calibrated": 0.009, "max_calibrated_probability": 0.339},
        "risk_tiers": [
            {"tier": "low", "share_of_cohort": 0.783},
            {"tier": "moderate", "share_of_cohort": 0.162},
            {"tier": "high", "share_of_cohort": 0.055},
        ],
        "subgroups": {
            "by_attribute": {
                "race": [
                    {
                        "group": "Caucasian",
                        "calibration_gap": 0.0,
                        "low_support": False,
                    },
                    {
                        "group": "Hispanic",
                        "calibration_gap": -0.02,
                        "low_support": False,
                    },
                    {"group": "Other", "calibration_gap": 0.032, "low_support": True},
                ]
            }
        },
    }


def failed_checks(metrics: dict) -> set[str]:
    return {name for name, passed, _ in quality_gate.check(metrics) if not passed}


def test_healthy_model_passes_every_check():
    assert failed_checks(healthy_metrics()) == set()


@pytest.mark.parametrize(
    ("mutate", "expected_failure"),
    [
        (lambda m: m["test_calibrated"].update(roc_auc=0.60), "ROC-AUC"),
        (
            lambda m: m["baselines"]["incumbent"].update(roc_auc=0.66),
            "Beats incumbent rule",
        ),
        (lambda m: m["lift_table"][1].update(lift=1.5), "Lift in top 10%"),
        (lambda m: m["calibration"].update(ece_calibrated=0.05), "Calibration error"),
        (
            lambda m: m["calibration"].update(max_calibrated_probability=1.0),
            "Highest predicted risk",
        ),
        (
            # 12 high + 10 moderate: total stays in range, only the high tier is off
            lambda m: (
                m["risk_tiers"][2].update(share_of_cohort=0.12),
                m["risk_tiers"][1].update(share_of_cohort=0.10),
            ),
            "High tier per 100",
        ),
        (
            lambda m: m["risk_tiers"][1].update(share_of_cohort=0.30),
            "High + moderate per 100",
        ),
        (
            lambda m: m["subgroups"]["by_attribute"]["race"][1].update(
                calibration_gap=-0.05
            ),
            "Subgroup calibration gap",
        ),
    ],
)
def test_each_regression_is_blocked(mutate, expected_failure):
    metrics = copy.deepcopy(healthy_metrics())
    mutate(metrics)
    assert failed_checks(metrics) == {expected_failure}


def test_low_support_groups_do_not_trip_the_gate():
    """'Other' has a 0.032 gap but too few readmissions to measure reliably."""
    metrics = healthy_metrics()
    assert "Subgroup calibration gap" not in failed_checks(metrics)


def test_main_exits_nonzero_when_a_check_fails(tmp_path, monkeypatch):
    metrics = healthy_metrics()
    metrics["test_calibrated"]["roc_auc"] = 0.55
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))
    monkeypatch.setattr(quality_gate, "METRICS_PATH", path)
    with pytest.raises(SystemExit, match="ROC-AUC"):
        quality_gate.main()


def test_main_succeeds_for_a_healthy_model(tmp_path, monkeypatch, capsys):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(healthy_metrics()))
    monkeypatch.setattr(quality_gate, "METRICS_PATH", path)
    quality_gate.main()
    assert "Quality gate passed" in capsys.readouterr().out
