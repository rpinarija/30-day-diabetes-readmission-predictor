"""Centralized configuration: paths, constants, and risk-tier thresholds."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "diabetic_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CALIBRATION_SIZE = 0.2

RISK_TIER_THRESHOLDS = {
    "moderate": 0.10,
    "high": 0.20,
}

def assign_risk_tier(probability: float) -> str:
    """Assign a risk tier based on predicted probability"""
    if probability >= RISK_TIER_THRESHOLDS["high"]:
        return "high"
    if probability >= RISK_TIER_THRESHOLDS["moderate"]:
        return "moderate"
    return "low"