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

from readmission.config import (
    CALIBRATION_METHOD,
    RISK_TIER_ORDER,
    RISK_TIER_THRESHOLDS,
    assign_risk_tier,
)

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
    flag_threshold: float = RISK_TIER_THRESHOLDS["moderate"],
    n_boot: int = 200,
    random_state: int = 0,
) -> list[dict]:
    """Per-subgroup calibration, ranking, and recall at a shared operating point.

    Patients are flagged with one probability cutoff for everyone - by default the
    moderate-tier threshold, i.e. anyone the tier policy sends to follow-up - so the
    recall figures describe the policy that would actually be deployed.
    Rows are ordered from largest to smallest group.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    groups = np.asarray(groups).astype(str)
    flagged = proba >= flag_threshold
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


def capacity_table(y_true, proba, cutoffs) -> list[dict]:
    """What following up with everyone at or above each risk cutoff would mean in practice.

    Volumes are per 100 discharges so they translate to any hospital's size.
    `number_needed_to_flag` is how many patients get follow-up per readmission
    among them.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    total_positives = max(int(y_true.sum()), 1)
    rows = []
    for cutoff in cutoffs:
        flagged = proba >= cutoff
        caught = int(y_true[flagged].sum())
        precision = caught / flagged.sum() if flagged.any() else None
        rows.append(
            {
                "cutoff": float(cutoff),
                "flagged_per_100": float(flagged.mean() * 100),
                "precision": None if precision is None else float(precision),
                "recall": float(caught / total_positives),
                "readmissions_flagged_per_100": float(caught / len(y_true) * 100),
                "number_needed_to_flag": (float(1 / precision) if precision else None),
            }
        )
    return rows


def decision_curve(y_true, proba, thresholds) -> list[dict]:
    """Net benefit of the model vs. following up with everyone, per threshold probability.

    Net benefit = TP/n - FP/n * t/(1-t). The threshold t encodes the trade-off: at
    t = 0.12, catching one readmission is judged worth ~7.3 unnecessary follow-ups
    (0.88 / 0.12). Following up with no one has net benefit 0 at every threshold.
    Requires calibrated probabilities - the cutoff is read as a real risk.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    n = len(y_true)
    prevalence = y_true.mean()
    rows = []
    for t in thresholds:
        odds = t / (1 - t)
        flagged = proba >= t
        true_pos = int((y_true[flagged] == 1).sum())
        false_pos = int((y_true[flagged] == 0).sum())
        rows.append(
            {
                "threshold": float(t),
                "net_benefit_model": float(true_pos / n - false_pos / n * odds),
                "net_benefit_treat_all": float(prevalence - (1 - prevalence) * odds),
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
        (f"Calibrated ({CALIBRATION_METHOD})", calibrated_proba, "#2A78D6"),
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


def decision_curve_plot(rows: list[dict], path: Path) -> None:
    """Net benefit of the model vs. follow up with everyone / no one, with tier cutoffs."""
    surface, ink, ink_muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    model_color, all_color = "#2a78d6", "#eb6834"  # categorical slots 1 and 2

    thresholds = [r["threshold"] for r in rows]
    model = [r["net_benefit_model"] for r in rows]
    treat_all = [r["net_benefit_treat_all"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.axhline(
        0, color=ink_muted, linewidth=1.5, linestyle="--", label="Follow up with no one"
    )
    ax.plot(
        thresholds,
        treat_all,
        color=all_color,
        linewidth=2,
        label="Follow up with everyone",
    )
    ax.plot(thresholds, model, color=model_color, linewidth=2, label="Model")
    # Cutoffs can sit close together, so stagger their labels vertically.
    for i, (tier, cutoff) in enumerate(RISK_TIER_THRESHOLDS.items()):
        ax.axvline(cutoff, color=grid, linewidth=1.5, zorder=0)
        ax.annotate(
            f"{tier} tier ≥ {cutoff:.2f}",
            (cutoff, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -14 - 14 * i),
            textcoords="offset points",
            color=ink_muted,
            fontsize=9,
        )

    top = max(model) * 1.25
    ax.set_ylim(-top * 0.25, top)
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_title(
        "Decision curve: when is the model worth using?", loc="left", color=ink, pad=12
    )
    ax.set_xlabel(
        "Threshold probability (risk at which follow-up is worthwhile)", color=ink_muted
    )
    ax.set_ylabel(
        "Net benefit (readmissions caught per patient, net of false alarms)",
        color=ink_muted,
    )
    ax.tick_params(colors=ink_muted, length=0)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.legend(loc="upper right", frameon=False, labelcolor=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=surface)
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
        (
            "Highest predicted risk: "
            f"{metrics['calibration'].get('max_calibrated_probability', float('nan')):.1%}"
        ),
        "",
        f"{'top k%':>8}{'flagged':>12}{'precision':>12}{'recall':>12}{'lift':>8}",
    ]
    for row in metrics["lift_table"]:
        lines.append(
            f"{row['top_fraction']:>8.0%}{row['n_flagged']:>12,}"
            f"{row['precision']:>12.3f}{row['recall']:>12.1%}{row['lift']:>7.2f}x"
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

    capacity = metrics.get("thresholds", {}).get("capacity")
    if capacity:
        lines.append("")
        lines.append(
            f"{'cutoff':>8}{'flagged/100':>13}{'precision':>11}{'recall':>9}"
            f"{'caught/100':>12}{'flagged per readmission':>25}"
        )
        for row in capacity:
            precision, nnf = row["precision"], row["number_needed_to_flag"]
            lines.append(
                f"{row['cutoff']:>8.2f}{row['flagged_per_100']:>13.1f}"
                f"{'n/a' if precision is None else f'{precision:.1%}':>11}"
                f"{row['recall']:>9.1%}{row['readmissions_flagged_per_100']:>12.2f}"
                f"{'n/a' if nnf is None else f'{nnf:.1f}':>25}"
            )

    subgroups = metrics.get("subgroups")
    if subgroups:
        flag = subgroups["flag_threshold"]
        for attribute, rows in subgroups["by_attribute"].items():
            lines.append("")
            lines.append(f"by {attribute} (flagged = predicted risk >= {flag:.0%})")
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
    "capacity_table",
    "decision_curve",
    "decision_curve_plot",
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
