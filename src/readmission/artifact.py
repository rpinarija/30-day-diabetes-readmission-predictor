"""The trained-model artifact: the model plus everything needed to serve it correctly.

Serving reads the tier thresholds and input ranges from the artifact, not from config, so
a deployed model always uses the values it was validated with - even if config changes
later without a retrain.
"""

from pathlib import Path

import joblib
import pandas as pd

from readmission.preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES

REQUIRED_KEYS = frozenset(
    {
        "model",
        "model_type",
        "model_version",
        "trained_at",
        "calibration_method",
        "risk_tier_thresholds",
        "feature_columns",
        "input_ranges",
        "metrics",
    }
)


def build_artifact(
    model,
    *,
    model_type: str,
    model_version: str,
    trained_at: str,
    calibration_method: str,
    risk_tier_thresholds: dict[str, float],
    X_train: pd.DataFrame,
    test_metrics: dict[str, float],
) -> dict:
    """Bundle a fitted, calibrated model with the metadata serving depends on."""
    return {
        "model": model,
        "model_type": model_type,
        "model_version": model_version,
        "trained_at": trained_at,
        "calibration_method": calibration_method,
        "risk_tier_thresholds": dict(risk_tier_thresholds),
        "feature_columns": list(FEATURE_COLUMNS),
        # Numeric ranges seen in training, so serving can flag extrapolation.
        "input_ranges": {
            column: [float(X_train[column].min()), float(X_train[column].max())]
            for column in NUMERIC_FEATURES
        },
        "metrics": {"test": test_metrics},
    }


def load_artifact(path: Path) -> dict:
    """Load and validate an artifact.

    joblib uses pickle, which can execute code: only load artifacts this project built.
    """
    artifact = joblib.load(path)
    missing = REQUIRED_KEYS - set(artifact)
    if missing:
        raise ValueError(
            f"Model artifact {path} is missing {sorted(missing)}; "
            "retrain with the current code."
        )
    if artifact["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(
            f"Model artifact {path} was trained on different features than this code "
            "produces; retrain before serving it."
        )
    return artifact
