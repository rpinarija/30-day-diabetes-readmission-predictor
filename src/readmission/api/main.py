"""Prediction service: scores one diabetic inpatient encounter at discharge.

Run locally:  uv run python -m uvicorn readmission.api.main:app --reload
Interactive docs are served at /docs.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request

from readmission import __version__
from readmission.api.schemas import Encounter, Health, ModelInfo, Prediction
from readmission.artifact import load_artifact
from readmission.config import MODEL_PATH, assign_risk_tier
from readmission.preprocessing import NUMERIC_FEATURES, engineer_features

MODEL_PATH_ENV = "READMISSION_MODEL_PATH"

INTENDED_USE = (
    "Prioritize discharge follow-up for adult diabetic inpatients. Not for denying or "
    "rationing care, coverage decisions, or individual diagnosis. See MODEL_CARD.md."
)


def range_warnings(features: pd.DataFrame, input_ranges: dict) -> list[str]:
    """Flag numeric features outside the range the model saw in training."""
    warnings = []
    row = features.iloc[0]
    for column in NUMERIC_FEATURES:
        low, high = input_ranges[column]
        value = row[column]
        if pd.notna(value) and not low <= value <= high:
            warnings.append(
                f"{column}={value:g} is outside the training range "
                f"{low:g}-{high:g}; the prediction is an extrapolation"
            )
    return warnings


def create_app(model_path: Path | None = None) -> FastAPI:
    """Build the app. The model path defaults to $READMISSION_MODEL_PATH, then config."""
    path = Path(model_path or os.environ.get(MODEL_PATH_ENV, MODEL_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast: if the model can't be loaded, the service doesn't start. A service
        # that reports itself up but can't predict hides a broken deployment.
        app.state.artifact = load_artifact(path)
        yield

    app = FastAPI(
        title="30-Day Readmission Risk",
        version=__version__,
        description=INTENDED_USE,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        return Health(
            status="ok", model_version=request.app.state.artifact["model_version"]
        )

    @app.get("/model-info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        artifact = request.app.state.artifact
        return ModelInfo(
            model_version=artifact["model_version"],
            model_type=artifact["model_type"],
            trained_at=artifact["trained_at"],
            calibration_method=artifact["calibration_method"],
            risk_tier_thresholds=artifact["risk_tier_thresholds"],
            test_metrics=artifact["metrics"]["test"],
            intended_use=INTENDED_USE,
        )

    # A plain `def` (not async): scoring is CPU-bound, so FastAPI runs it in a worker
    # thread instead of blocking the event loop.
    @app.post("/predict", response_model=Prediction)
    def predict(encounter: Encounter, request: Request) -> Prediction:
        artifact = request.app.state.artifact
        features = engineer_features(pd.DataFrame([encounter.to_raw_record()]))
        risk = float(artifact["model"].predict_proba(features)[0, 1])
        return Prediction(
            readmission_risk=risk,
            risk_tier=assign_risk_tier(risk, artifact["risk_tier_thresholds"]),
            warnings=range_warnings(features, artifact["input_ranges"]),
            model_version=artifact["model_version"],
        )

    return app


app = create_app()
