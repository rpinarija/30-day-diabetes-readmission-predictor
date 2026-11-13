"""API tests against a small model trained on synthetic data, so they run without the
real dataset. The most important check: the API scores exactly like the training path."""

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from readmission.api.main import create_app
from readmission.api.schemas import Encounter
from readmission.artifact import build_artifact, load_artifact
from readmission.config import assign_risk_tier
from readmission.preprocessing import (
    build_preprocessing_pipeline,
    engineer_features,
    make_binary_target,
)
from tests.conftest import make_raw_encounter, make_synthetic_cohort

TEST_THRESHOLDS = {"moderate": 0.3, "high": 0.6}


def encounter_payload(**overrides) -> dict:
    """An API request equivalent to make_raw_encounter()'s defaults."""
    payload = {
        "age": 75,
        "gender": "Female",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 1,
        "time_in_hospital": 5,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
        "number_diagnoses": 5,
        "medical_specialty": "InternalMedicine",
        "diag_1": "428",
        "diag_2": "215.02",
        "diag_3": "401",
        "a1c_result": ">8",
        "max_glucose_serum": None,
        "medications": {"insulin": "Up", "metformin": "Steady"},
        "diabetes_medication_changed": True,
    }
    payload.update(overrides)
    return payload


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory):
    raw = make_synthetic_cohort()
    X, y = engineer_features(raw), make_binary_target(raw["readmitted"])
    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            (
                "model",
                XGBClassifier(
                    n_estimators=40,
                    max_depth=3,
                    learning_rate=0.3,
                    random_state=0,
                    n_jobs=1,
                ),
            ),
        ]
    ).fit(X.iloc[:300], y.iloc[:300])
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(pipeline), method="sigmoid"
    ).fit(X.iloc[300:], y.iloc[300:])
    artifact = build_artifact(
        calibrated,
        model_type="xgboost",
        model_version="test-version",
        trained_at="2026-01-01T00:00:00+00:00",
        calibration_method="sigmoid",
        risk_tier_thresholds=TEST_THRESHOLDS,
        X_train=X.iloc[:300],
        test_metrics={"roc_auc": 0.9},
    )
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(artifact, path)
    return path


@pytest.fixture(scope="module")
def client(artifact_path):
    with TestClient(create_app(artifact_path)) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health and metadata
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test-version"}


def test_model_info_reports_the_artifacts_thresholds(client):
    info = client.get("/model-info").json()
    assert info["risk_tier_thresholds"] == TEST_THRESHOLDS
    assert info["calibration_method"] == "sigmoid"
    assert "MODEL_CARD.md" in info["intended_use"]


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class TestPredict:
    def test_valid_encounter_is_scored(self, client):
        response = client.post("/predict", json=encounter_payload())
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["readmission_risk"] <= 1.0
        assert body["risk_tier"] in {"low", "moderate", "high"}
        assert body["model_version"] == "test-version"
        assert body["warnings"] == []

    def test_request_produces_the_same_features_as_training(self):
        """No training/serving skew: an API request and the equivalent raw dataset row
        must produce identical model inputs."""
        from_api = engineer_features(
            pd.DataFrame([Encounter(**encounter_payload()).to_raw_record()])
        )
        from_training = engineer_features(pd.DataFrame([make_raw_encounter()]))
        pd.testing.assert_frame_equal(from_api, from_training, check_dtype=False)

    def test_risk_matches_calling_the_model_directly(self, client, artifact_path):
        model = load_artifact(artifact_path)["model"]
        raw = pd.DataFrame([make_raw_encounter(number_inpatient=4)])
        expected = model.predict_proba(engineer_features(raw))[0, 1]
        body = client.post(
            "/predict", json=encounter_payload(number_inpatient=4)
        ).json()
        assert body["readmission_risk"] == pytest.approx(expected, abs=1e-12)

    def test_planted_signal_raises_risk(self, client):
        low = client.post("/predict", json=encounter_payload(number_inpatient=0)).json()
        high = client.post(
            "/predict", json=encounter_payload(number_inpatient=5)
        ).json()
        assert high["readmission_risk"] > low["readmission_risk"]

    def test_tier_follows_the_artifacts_thresholds(self, client):
        for inpatient in range(6):
            body = client.post(
                "/predict", json=encounter_payload(number_inpatient=inpatient)
            ).json()
            assert body["risk_tier"] == assign_risk_tier(
                body["readmission_risk"], TEST_THRESHOLDS
            )

    def test_thresholds_come_from_the_artifact_not_config(
        self, artifact_path, tmp_path
    ):
        """Unreachable thresholds in the artifact -> every patient is low tier, even a
        high-risk one that config's thresholds would call high."""
        artifact = load_artifact(artifact_path)
        artifact["risk_tier_thresholds"] = {"moderate": 2.0, "high": 3.0}
        path = tmp_path / "model.joblib"
        joblib.dump(artifact, path)
        with TestClient(create_app(path)) as other_client:
            body = other_client.post(
                "/predict", json=encounter_payload(number_inpatient=5)
            ).json()
        assert body["risk_tier"] == "low"
        assert assign_risk_tier(body["readmission_risk"]) != "low"

    def test_optional_fields_can_be_omitted(self, client):
        payload = encounter_payload()
        for field in (
            "medical_specialty",
            "diag_1",
            "diag_2",
            "diag_3",
            "a1c_result",
            "max_glucose_serum",
            "medications",
        ):
            payload.pop(field)
        assert client.post("/predict", json=payload).status_code == 200

    def test_out_of_range_input_is_warned(self, client):
        body = client.post(
            "/predict", json=encounter_payload(number_emergency=40)
        ).json()
        assert any("number_emergency" in w for w in body["warnings"])


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"age": 15},  # under 18 is out of scope
            {"time_in_hospital": 15},  # training covered 1-14 days
            {"number_inpatient": -1},
            {"gender": "Unknown"},
            {"admission_type_id": 99},
            {"medications": {"aspirin": "Up"}},
            {"medications": {"insulin": "Maybe"}},
            {"a1c_result": "high"},
            {"race": "Caucasian"},  # not a model input; unknown fields are rejected
        ],
    )
    def test_invalid_requests_are_rejected(self, client, overrides):
        response = client.post("/predict", json=encounter_payload(**overrides))
        assert response.status_code == 422

    def test_missing_required_field_is_rejected(self, client):
        payload = encounter_payload()
        payload.pop("diabetes_medication_changed")
        assert client.post("/predict", json=payload).status_code == 422

    def test_hospice_discharge_explains_why(self, client):
        response = client.post(
            "/predict", json=encounter_payload(discharge_disposition_id=13)
        )
        assert response.status_code == 422
        assert "out of scope" in response.text

    @pytest.mark.parametrize("code", ["E11.9", "I50.9", "J18.9"])
    def test_icd10_codes_are_rejected_with_explanation(self, client, code):
        response = client.post("/predict", json=encounter_payload(diag_1=code))
        assert response.status_code == 422
        assert "ICD-10" in response.text

    @pytest.mark.parametrize("code", ["428", "250.83", "V45", "V57.1", "E909", "8"])
    def test_icd9_codes_are_accepted(self, client, code):
        assert (
            client.post("/predict", json=encounter_payload(diag_1=code)).status_code
            == 200
        )


class TestToRawRecord:
    @pytest.mark.parametrize(
        ("age", "bucket"),
        [(18, "[10-20)"), (75, "[70-80)"), (99, "[90-100)"), (104, "[90-100)")],
    )
    def test_age_maps_to_dataset_bucket(self, age, bucket):
        assert Encounter(**encounter_payload(age=age)).to_raw_record()["age"] == bucket

    def test_diabetes_med_is_derived_from_medications(self):
        on_meds = Encounter(**encounter_payload()).to_raw_record()
        off_meds = Encounter(**encounter_payload(medications={})).to_raw_record()
        assert on_meds["diabetesMed"] == "Yes"
        assert off_meds["diabetesMed"] == "No"


# ---------------------------------------------------------------------------
# Startup and artifact checks
# ---------------------------------------------------------------------------


class TestStartup:
    def test_missing_model_fails_fast(self, tmp_path):
        with (
            pytest.raises(FileNotFoundError),
            TestClient(create_app(tmp_path / "missing.joblib")),
        ):
            pass

    def test_artifact_missing_metadata_is_rejected(self, artifact_path, tmp_path):
        artifact = load_artifact(artifact_path)
        del artifact["risk_tier_thresholds"]
        path = tmp_path / "old.joblib"
        joblib.dump(artifact, path)
        with pytest.raises(ValueError, match="missing"):
            load_artifact(path)

    def test_artifact_trained_on_other_features_is_rejected(
        self, artifact_path, tmp_path
    ):
        """e.g. an old model that still used race as an input."""
        artifact = load_artifact(artifact_path)
        artifact["feature_columns"] = artifact["feature_columns"] + ["race"]
        path = tmp_path / "stale.joblib"
        joblib.dump(artifact, path)
        with pytest.raises(ValueError, match="different features"):
            load_artifact(path)
