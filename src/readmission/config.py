"""Centralized configuration: paths, constants, and risk-tier thresholds."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "diabetic_data.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
CALIBRATION_PLOT_PATH = MODEL_DIR / "reliability.png"
FEATURE_IMPORTANCE_PLOT_PATH = MODEL_DIR / "feature_importance.png"
DECISION_CURVE_PLOT_PATH = MODEL_DIR / "decision_curve.png"
TUNING_LOG_PATH = MODEL_DIR / "tuning_runs.jsonl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CALIBRATION_SIZE = 0.2

# Sigmoid (Platt) rather than isotonic calibration. Class weighting mostly shifts the model's
# log-odds, which a sigmoid corrects directly. Isotonic fit a step function whose top steps
# rested on a handful of calibration patients (one step predicted 100% risk from a single
# patient); sigmoid matched its calibration error with better log loss, no ties, and a
# highest predicted risk of ~35%.
CALIBRATION_METHOD = "sigmoid"

RISK_TIER_ORDER = ("low", "moderate", "high")

# Calibrated-probability cutoffs, sized to follow-up capacity rather than picked as round
# numbers. On the calibration split: high (>= 0.17) flags ~6 per 100 discharges with ~21%
# readmitted, an intensive-follow-up list; high + moderate (>= 0.12) flags ~22 per 100 and
# catches ~40% of readmissions. Both cutoffs sit where decision-curve net benefit beats
# following up with everyone or no one. Cutoffs depend on the calibration method and on
# each trained model: re-check flagged volumes after every retrain.
RISK_TIER_THRESHOLDS = {
    "moderate": 0.12,
    "high": 0.17,
}


def assign_risk_tier(
    probability: float, thresholds: dict[str, float] | None = None
) -> str:
    """Assign a risk tier based on predicted probability.

    Serving passes the thresholds stored with the model; training uses the config values.
    """
    thresholds = RISK_TIER_THRESHOLDS if thresholds is None else thresholds
    if probability >= thresholds["high"]:
        return "high"
    if probability >= thresholds["moderate"]:
        return "moderate"
    return "low"
