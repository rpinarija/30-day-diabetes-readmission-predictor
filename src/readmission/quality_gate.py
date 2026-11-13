"""Release gate: checks a newly trained model must pass before it is published.

Run after training:  uv run python -m readmission.quality_gate
Exits non-zero if any check fails, so CI never publishes a broken or degraded model.

Limits leave headroom around the current model (test ROC-AUC 0.649, lift 2.18x, ECE
0.009, highest risk 0.34, high tier 5.5 and high + moderate 21.7 per 100 discharges).
They are meant to catch a broken pipeline or a real regression, not run-to-run noise.
The tier-volume ranges are the model card's monitoring thresholds.
"""

import json
import sys

from readmission.config import METRICS_PATH

MIN_ROC_AUC = 0.63
MIN_LIFT_AT_10PCT = 1.9
MAX_ECE = 0.02
MAX_PREDICTED_RISK = 0.60
HIGH_TIER_PER_100 = (4.0, 8.0)
FLAGGED_PER_100 = (18.0, 26.0)  # high + moderate tiers
MAX_SUBGROUP_CALIBRATION_GAP = 0.03  # groups with enough readmissions to measure


def check(metrics: dict) -> list[tuple[str, bool, str]]:
    """Every gate check as (name, passed, detail)."""
    model = metrics["test_calibrated"]
    incumbent = metrics["baselines"]["incumbent"]
    lift_10 = next(r for r in metrics["lift_table"] if r["top_fraction"] == 0.10)
    calibration = metrics["calibration"]
    tiers = {r["tier"]: r["share_of_cohort"] * 100 for r in metrics["risk_tiers"]}
    high, flagged = tiers["high"], tiers["high"] + tiers["moderate"]
    worst_gap, worst_group = max(
        (abs(row["calibration_gap"]), f"{attribute}={row['group']}")
        for attribute, rows in metrics["subgroups"]["by_attribute"].items()
        for row in rows
        if not row["low_support"]
    )

    return [
        (
            "ROC-AUC",
            model["roc_auc"] >= MIN_ROC_AUC,
            f"{model['roc_auc']:.4f} (min {MIN_ROC_AUC})",
        ),
        (
            "Beats incumbent rule",
            model["roc_auc"] > incumbent["roc_auc"],
            f"{model['roc_auc']:.4f} vs {incumbent['roc_auc']:.4f}",
        ),
        (
            "Lift in top 10%",
            lift_10["lift"] >= MIN_LIFT_AT_10PCT,
            f"{lift_10['lift']:.2f}x (min {MIN_LIFT_AT_10PCT}x)",
        ),
        (
            "Calibration error",
            calibration["ece_calibrated"] <= MAX_ECE,
            f"{calibration['ece_calibrated']:.4f} (max {MAX_ECE})",
        ),
        (
            "Highest predicted risk",
            calibration["max_calibrated_probability"] <= MAX_PREDICTED_RISK,
            f"{calibration['max_calibrated_probability']:.3f} (max {MAX_PREDICTED_RISK})",
        ),
        (
            "High tier per 100",
            HIGH_TIER_PER_100[0] <= high <= HIGH_TIER_PER_100[1],
            f"{high:.1f} (range {HIGH_TIER_PER_100[0]:g}-{HIGH_TIER_PER_100[1]:g})",
        ),
        (
            "High + moderate per 100",
            FLAGGED_PER_100[0] <= flagged <= FLAGGED_PER_100[1],
            f"{flagged:.1f} (range {FLAGGED_PER_100[0]:g}-{FLAGGED_PER_100[1]:g})",
        ),
        (
            "Subgroup calibration gap",
            worst_gap <= MAX_SUBGROUP_CALIBRATION_GAP,
            f"{worst_gap:.3f} at {worst_group} (max {MAX_SUBGROUP_CALIBRATION_GAP})",
        ),
    ]


def main() -> None:
    results = check(json.loads(METRICS_PATH.read_text()))
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<26}{detail}")
    failed = [name for name, passed, _ in results if not passed]
    if failed:
        sys.exit(f"\nQuality gate failed: {', '.join(failed)}. Model not published.")
    print("\nQuality gate passed.")


if __name__ == "__main__":
    main()
