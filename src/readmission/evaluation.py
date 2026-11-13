"""Evaluation metrics, reference baselines, and calibration diagnostics for readmission prediction models."""

from itertools import pairwise
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from readmission.config import RISK_TIER_ORDER, RISK_TIER_THRESHOLDS, assign_risk_tier

matplotlib.use("Agg")

DEFAULT_TOP_PROPORTIONS = (0.05, 0.10, 0.15, 0.20, 0.30)

# Age bands for subgroup evaluation. Wider than the raw 10-year buckets so each band
# has enough readmissions in the test set for stable estimates.
AGE_BAND_EDGES = [0, 50, 70, 80, 100]
AGE_BAND_LABELS = ["<50", "50-69", "70-79", "80+"]

# Below this many readmitted patients, a subgroup's ROC-AUC is too noisy to read much into.
MIN_SUBGROUP_POSITIVES = 30


def evaluate(y_true, proba) -> dict[str, float]:
    """Headlines scores for a set of predicted probabilities."""
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba)),
    }


def no_skill_baselines(y_true) -> dict[str, float]:
    """Reference baseline scores for a no-skill model"""
    p = float(np.mean(y_true))
    return {
        "prevalence": p,
        "accuracy": max(p, 1.0 - p),
        "roc_auc": 0.5,
        "pr_auc": p,
        "brier": p * (1.0 - p),
        "log_loss": float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))),
    }


def incumbent_baseline(
    X: pd.DataFrame, y_true, column: str = "number_inpatient"
) -> dict[str, float]:
    """Reference baseline scores for an incumbent model"""
    scores = X[column].to_numpy(dtype=float)
    result = {
        "feature": column,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }
    top_10 = lift_table(y_true, scores, fractions=(0.10,))[0]
    result["precision_top_10pct"] = top_10["precision"]
    result["list_top_10pct"] = top_10["n_flagged"]
    result["lift_at_10pct"] = top_10["lift"]
    return result


def lift_table(
    y_true, proba, fractions: tuple[float, ...] = DEFAULT_TOP_PROPORTIONS
) -> list[dict[str, float]]:
    """Lift table for a set of predicted probabilities."""
    proba = np.asarray(proba, dtype=float)
    y_sorted = np.asarray(y_true)[np.argsort(-proba)]
    prevalence = y_sorted.mean()
    total_positives = max(int(y_sorted.sum()), 1)

    rows = []
    for fraction in fractions:
        n = max(int(len(y_sorted) * fraction), 1)
        top_k_positives = int(y_sorted[:n].sum())
        precision = top_k_positives / n
        rows.append(
            {
                "top_fraction": float(fraction),
                "n_flagged": n,
                "precision": float(precision),
                "recall": float(top_k_positives / total_positives),
                "lift": float(precision / prevalence) if prevalence else float("nan"),
            }
        )
    return rows


def _quantile_bins(proba: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Return the bin index for each probability based on quantiles."""
    edges = np.quantile(proba, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    masks = []
    for low, high in pairwise(edges):
        if high <= low:
            continue
        mask = (proba >= low) & (proba < high)
        if mask.any():
            masks.append(mask)
    return masks


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Expected calibration error (ECE) for a set of predicted probabilities."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    error = sum(
        mask.sum() / len(proba) * abs(y_true[mask].mean() - proba[mask].mean())
        for mask in _quantile_bins(proba, n_bins)
    )
    return float(error)


def murphy_decomposition(y_true, proba, n_bins: int = 10) -> dict[str, float]:
    """Decompose the Murphy score into its components: reliability, resolution, and uncertainty."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    base_rate = y_true.mean()

    reliability = resolution = 0.0
    for mask in _quantile_bins(proba, n_bins):
        weight = mask.sum() / len(proba)
        observed = y_true[mask].mean()
        reliability += weight * (proba[mask].mean() - observed) ** 2
        resolution += weight * (base_rate - observed) ** 2

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(base_rate * (1 - base_rate)),
    }


def risk_tier_summary(y_true, proba) -> list[dict[str, float]]:
    """Patients per risk tier and their true readmission rates."""
    y_true = np.asarray(y_true)
    tiers = np.array([assign_risk_tier(float(p)) for p in np.asarray(proba)])
    rows = []
    for tier in RISK_TIER_ORDER:
        mask = tiers == tier
        rows.append(
            {
                "tier": tier,
                "n_patients": int(mask.sum()),
                "share_of_cohort": float(mask.mean()),
                "observed_readmission_rate": (
                    float(y_true[mask].mean()) if mask.any() else None
                ),
            }
        )
    return rows


def age_band(age_midpoint: pd.Series) -> pd.Series:
    """Bucket age midpoints into the bands used for subgroup evaluation."""
    bands = pd.cut(
        age_midpoint, bins=AGE_BAND_EDGES, labels=AGE_BAND_LABELS, right=False
    )
    return bands.astype(str)


def subgroup_columns(X: pd.DataFrame) -> dict[str, pd.Series]:
    """The patient attributes to break performance out by, one label per patient."""
    return {
        "race": X["race"],
        "gender": X["gender"],
        "age_band": age_band(X["age_midpoint"]),
    }


def _bootstrap_auc_ci(
    y_true: np.ndarray, proba: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float] | tuple[None, None]:
    """95% percentile-bootstrap interval for ROC-AUC."""
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue  # resample drew a single class; AUC undefined
        aucs.append(roc_auc_score(y_true[idx], proba[idx]))
    if len(aucs) < n_boot // 2:
        return None, None
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def subgroup_metrics(
    y_true,
    proba,
    groups,
    flag_fraction: float = 0.10,
    n_boot: int = 200,
    random_state: int = 0,
) -> list[dict]:
    """Per-subgroup calibration, ranking, and recall at a shared operating point.

    Patients are flagged using one cutoff for everyone (the top `flag_fraction` of the
    whole cohort), mirroring how a care team would work a single prioritized list.
    Rows are ordered from largest to smallest group.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    groups = np.asarray(groups).astype(str)
    flagged = proba >= np.quantile(proba, 1 - flag_fraction)
    rng = np.random.default_rng(random_state)

    values, counts = np.unique(groups, return_counts=True)
    rows = []
    for value in values[np.argsort(-counts, kind="stable")]:
        mask = groups == value
        y_g, p_g, flagged_g = y_true[mask], proba[mask], flagged[mask]
        n_positives = int(y_g.sum())
        has_both_classes = 0 < n_positives < mask.sum()
        ci_low, ci_high = (
            _bootstrap_auc_ci(y_g, p_g, n_boot, rng)
            if has_both_classes
            else (None, None)
        )
        rows.append(
            {
                "group": str(value),
                "n_patients": int(mask.sum()),
                "n_readmitted": n_positives,
                "observed_rate": float(y_g.mean()),
                "mean_predicted": float(p_g.mean()),
                "calibration_gap": float(p_g.mean() - y_g.mean()),
                "roc_auc": float(roc_auc_score(y_g, p_g)) if has_both_classes else None,
                "roc_auc_ci": [ci_low, ci_high],
                "flag_rate": float(flagged_g.mean()),
                "recall_at_flag": (
                    float(flagged_g[y_g == 1].mean()) if n_positives else None
                ),
                "low_support": n_positives < MIN_SUBGROUP_POSITIVES,
            }
        )
    return rows


def calibration_curve_plot(
    y_true, uncalibrated_proba, calibrated_proba, path: Path, n_bins: int = 10
) -> None:
    """Calibration curve for a set of predicted probabilities."""

    fig, (curve_ax, hist_ax) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    curve_ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibrated")
    for label, proba, color in (
        ("Uncalibrated", uncalibrated_proba, "#D03B3B"),
        ("Calibrated (isotonic)", calibrated_proba, "#2A78D6"),
    ):
        observed, predicted = calibration_curve(
            y_true, proba, n_bins=n_bins, strategy="quantile"
        )
        curve_ax.plot(predicted, observed, marker="o", label=label, color=color)
        hist_ax.hist(proba, bins=40, alpha=0.5, label=label, color=color)

    upper = max(np.max(uncalibrated_proba), np.max(calibrated_proba)) * 1.05
    curve_ax.set(
        xlim=(0, upper),
        ylim=(0, 1),
        ylabel="Observed Readmission Rate",
        title="Calibration Curve",
    )
    curve_ax.legend(loc="upper right")
    curve_ax.grid(alpha=0.3)
    hist_ax.set(
        xlim=(0, upper), xlabel="Predicted Probability", ylabel="Count", yscale="log"
    )
    hist_ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def format_summary(metrics: dict) -> str:
    """Format a summary of evaluation metrics - every score next to its reference point."""
    ns = metrics["baselines"]["no_skill"]
    cal = metrics["test_calibrated"]
    inc = metrics["baselines"]["incumbent"]
    lines = [
        "",
        f"{'METRIC':<12}{'no-skill':>12}{'incumbent':>12}{'model':>12}",
        f"{'roc_auc':<12}{ns['roc_auc']:>12.4f}{inc['roc_auc']:>12.4f}{cal['roc_auc']:>12.4f}",
        f"{'pr_auc':<12}{ns['pr_auc']:>12.4f}{inc['pr_auc']:>12.4f}{cal['pr_auc']:>12.4f}",
        f"{'brier':<12}{ns['brier']:>12.4f}{'-':>12}{cal['brier']:>12.4f}",
        f"{'log_loss':<12}{ns['log_loss']:>12.4f}{'-':>12}{cal['log_loss']:>12.4f}",
        "",
        f"Brier score vs base rate: {1 - cal['brier'] / ns['brier']:.1%}",
        (
            "Calibration error (ECE): "
            f"{metrics['calibration']['ece_uncalibrated']:.4f} -> "
            f"{metrics['calibration']['ece_calibrated']:.4f}"
        ),
        "",
        f"{'top k%':>8}{'flagged':>12}{'precision':>12}{'recall':>12}{'lift':>8}",
    ]
    for row in metrics["lift_table"]:
        lines.append(
            f"{row['top_fraction']:>8.0%}{row['n_flagged']:>10}"
            f"{row['precision']:>12.3f}{row['recall']:>10.1f}{row['lift']:>8.2f}x"
        )
    lines.append("")
    lines.append(f"{'risk_tier':>10}{'patients':>12}{'share':>10}{'observed rate':>16}")
    for row in metrics["risk_tiers"]:
        observed = row["observed_readmission_rate"]
        rendered = f"{observed:.1%}" if observed is not None else "n/a"
        lines.append(
            f"{row['tier']:>10}{row['n_patients']:>12,}"
            f"{row['share_of_cohort']:>10.1%}{rendered:>16}"
        )

    subgroups = metrics.get("subgroups")
    if subgroups:
        flag = subgroups["flag_fraction"]
        for attribute, rows in subgroups["by_attribute"].items():
            lines.append("")
            lines.append(f"by {attribute} (flagged = top {flag:.0%} of all patients)")
            lines.append(
                f"{'group':>18}{'patients':>10}{'observed':>10}{'predicted':>11}"
                f"{'roc_auc (95% CI)':>24}{'recall@flag':>13}"
            )
            for row in rows:
                low, high = row["roc_auc_ci"]
                if row["roc_auc"] is None:
                    auc = "n/a"
                elif low is None:
                    auc = f"{row['roc_auc']:.3f}"
                else:
                    auc = f"{row['roc_auc']:.3f} ({low:.2f}-{high:.2f})"
                recall = row["recall_at_flag"]
                name = row["group"] + ("*" if row["low_support"] else "")
                lines.append(
                    f"{name:>18}{row['n_patients']:>10,}{row['observed_rate']:>10.1%}"
                    f"{row['mean_predicted']:>11.1%}{auc:>24}"
                    f"{'n/a' if recall is None else f'{recall:.1%}':>13}"
                )
        lines.append(
            f"* fewer than {MIN_SUBGROUP_POSITIVES} readmissions: metrics are noisy"
        )
    return "\n".join(lines)


__all__ = [
    "RISK_TIER_ORDER",
    "RISK_TIER_THRESHOLDS",
    "age_band",
    "calibration_curve_plot",
    "evaluate",
    "expected_calibration_error",
    "format_summary",
    "incumbent_baseline",
    "lift_table",
    "murphy_decomposition",
    "no_skill_baselines",
    "risk_tier_summary",
    "subgroup_columns",
    "subgroup_metrics",
]
