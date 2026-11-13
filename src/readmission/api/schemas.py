"""Request and response contracts for the prediction API.

The request uses readable field names and is translated into the raw dataset's columns by
`Encounter.to_raw_record()`, so serving runs exactly the same feature engineering as
training. Validation limits come from the training data and the model card's scope.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from readmission.preprocessing import (
    EXPIRED_OR_HOSPICE_DISPOSITIONS,
    MEDICATION_COLUMNS,
)

# ICD-9-CM: numeric codes (leading zeros may be dropped), V-codes, and E-codes. Every
# diagnosis code in the training data matches; ICD-10 codes (e.g. "E11.9") do not.
ICD9_PATTERN = re.compile(
    r"^(\d{1,3}(\.\d{1,2})?|V\d{2}(\.\d{1,2})?|E\d{3}(\.\d{1,2})?)$"
)

MedicationStatus = Literal["No", "Steady", "Up", "Down"]

EXAMPLE_ENCOUNTER = {
    "age": 72,
    "gender": "Female",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 5,
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 16,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "number_diagnoses": 9,
    "medical_specialty": "InternalMedicine",
    "diag_1": "428",
    "diag_2": "250.02",
    "diag_3": "401",
    "a1c_result": ">8",
    "max_glucose_serum": None,
    "medications": {"insulin": "Up", "metformin": "Steady"},
    "diabetes_medication_changed": True,
}


class Encounter(BaseModel):
    """One diabetic inpatient encounter, scored at discharge."""

    # Unknown fields are rejected rather than ignored, so a misspelled field name
    # (e.g. "a1c" for "a1c_result") fails loudly instead of being silently dropped.
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [EXAMPLE_ENCOUNTER]}
    )

    age: int = Field(
        ge=18, le=120, description="Age in years. Under 18 is out of scope."
    )
    gender: Literal["Female", "Male"]
    admission_type_id: int = Field(ge=1, le=8, description="Code from IDS_mapping.csv.")
    discharge_disposition_id: int = Field(
        ge=1, le=30, description="Code from IDS_mapping.csv."
    )
    admission_source_id: int = Field(
        ge=1, le=26, description="Code from IDS_mapping.csv."
    )
    time_in_hospital: int = Field(
        ge=1,
        le=14,
        description="Length of stay in days; training covered 1-14 days only.",
    )
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=0)
    number_outpatient: int = Field(
        ge=0, description="Outpatient visits in the prior year."
    )
    number_emergency: int = Field(
        ge=0, description="Emergency visits in the prior year."
    )
    number_inpatient: int = Field(
        ge=0, description="Inpatient stays in the prior year."
    )
    number_diagnoses: int = Field(ge=1)
    medical_specialty: str | None = Field(
        default=None, max_length=100, description="Admitting physician's specialty."
    )
    diag_1: str | None = Field(
        default=None, description="Primary diagnosis (ICD-9-CM)."
    )
    diag_2: str | None = Field(
        default=None, description="Secondary diagnosis (ICD-9-CM)."
    )
    diag_3: str | None = Field(
        default=None, description="Additional diagnosis (ICD-9-CM)."
    )
    a1c_result: Literal["Norm", ">7", ">8"] | None = Field(
        default=None, description="HbA1c result; null if not tested."
    )
    max_glucose_serum: Literal["Norm", ">200", ">300"] | None = Field(
        default=None, description="Max glucose serum result; null if not tested."
    )
    medications: dict[str, MedicationStatus] = Field(
        default_factory=dict,
        description="Status of each diabetes medication; omitted ones count as 'No'.",
    )
    diabetes_medication_changed: bool = Field(
        description="Whether diabetes medications changed during the stay (dose or drug)."
    )

    @field_validator("discharge_disposition_id")
    @classmethod
    def _in_scope_disposition(cls, value: int) -> int:
        if value in EXPIRED_OR_HOSPICE_DISPOSITIONS:
            raise ValueError(
                "patients who died or were discharged to hospice are out of scope "
                "(they cannot be readmitted)"
            )
        return value

    @field_validator("diag_1", "diag_2", "diag_3")
    @classmethod
    def _icd9_code(cls, value: str | None) -> str | None:
        if value is not None and not ICD9_PATTERN.match(value):
            raise ValueError(
                "must be an ICD-9-CM code (e.g. '428', '250.02', 'V45'); "
                "ICD-10 codes are not supported - see the model card"
            )
        return value

    @field_validator("medications")
    @classmethod
    def _known_medications(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(value) - set(MEDICATION_COLUMNS))
        if unknown:
            raise ValueError(
                f"unknown medications {unknown}; expected names from {MEDICATION_COLUMNS}"
            )
        return value

    def to_raw_record(self) -> dict:
        """Translate into the raw dataset's column names and value formats."""
        decade = min(self.age // 10 * 10, 90)  # 100+ falls in the dataset's top bucket
        medications = {
            med: self.medications.get(med, "No") for med in MEDICATION_COLUMNS
        }
        return {
            "age": f"[{decade}-{decade + 10})",
            "gender": self.gender,
            "admission_type_id": self.admission_type_id,
            "discharge_disposition_id": self.discharge_disposition_id,
            "admission_source_id": self.admission_source_id,
            "time_in_hospital": self.time_in_hospital,
            "num_lab_procedures": self.num_lab_procedures,
            "num_procedures": self.num_procedures,
            "num_medications": self.num_medications,
            "number_outpatient": self.number_outpatient,
            "number_emergency": self.number_emergency,
            "number_inpatient": self.number_inpatient,
            "number_diagnoses": self.number_diagnoses,
            "medical_specialty": self.medical_specialty,
            "diag_1": self.diag_1,
            "diag_2": self.diag_2,
            "diag_3": self.diag_3,
            "A1Cresult": self.a1c_result,
            "max_glu_serum": self.max_glucose_serum,
            **medications,
            "change": "Ch" if self.diabetes_medication_changed else "No",
            # In the training data diabetesMed is "Yes" exactly when any diabetes
            # medication is active (100% of rows), so derive it instead of asking for it.
            "diabetesMed": "Yes"
            if any(s != "No" for s in medications.values())
            else "No",
        }


class Factor(BaseModel):
    feature: str = Field(description="Model feature name.")
    label: str = Field(description="Plain-language name for care teams.")
    value: str | int | float = Field(
        description="This patient's value for the feature."
    )
    effect: Literal["raises risk", "lowers risk"]
    log_odds_contribution: float = Field(
        description=(
            "SHAP contribution to the pre-calibration log-odds. Use it to rank factors, "
            "not as percentage points of risk."
        )
    )


class Prediction(BaseModel):
    readmission_risk: float = Field(
        description="Calibrated probability of readmission within 30 days."
    )
    risk_tier: Literal["low", "moderate", "high"]
    top_factors: list[Factor] = Field(
        description=(
            "The features that moved this patient's score the most, largest first. "
            "They describe the model's reasoning, not clinical causes."
        )
    )
    warnings: list[str] = Field(
        description="Inputs outside the training data's range; the prediction extrapolates."
    )
    model_version: str


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str


class ModelInfo(BaseModel):
    model_version: str
    model_type: str
    trained_at: str
    calibration_method: str
    risk_tier_thresholds: dict[str, float]
    test_metrics: dict[str, float]
    intended_use: str
