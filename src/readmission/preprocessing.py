"""Cleaning and feature engineering for the UCI diabetes readmission dataset."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age_midpoint",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "service_utilization",
    "n_medication_changes",
    "n_active_medications",
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "admission_type",
    "discharge_disposition",
    "admission_source",
    "medical_specialty_group",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "metformin",
    "change",
    "diabetesMed",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# All 23 diabetes-medications columns in the dataset
# Possible values per column are: 'No', 'Steady', 'Up', 'Down'
MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Discharge disposition codes that indicate the patient has died or is in hospice care
# These labels cannot be readmitted, so we will exclude them from the dataset
EXPIRED_OR_HOSPICE_DISPOSITIONS = frozenset({11, 13, 14, 19, 20, 21})

# Mapping of age ranges to their midpoints for numerical representation
# Numeric values help preserve the model's ability to learn from the progression of age
AGE_MIDPOINTS = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}

# ----------------------------------------------------------------------------------------
# Target
# ----------------------------------------------------------------------------------------

def make_binary_target(readmitted: pd.Series) -> pd.Series:
    """
    Convert the readmitted column to a binary target variable.
    1 if readmitted within 30 days, 0 otherwise.
    """
    return (readmitted == "<30").astype(int)


# ----------------------------------------------------------------------------------------
# Row-level mapping functions
# ----------------------------------------------------------------------------------------

def map_age_to_midpoint(age: pd.Series) -> pd.Series:
    return age.map(AGE_MIDPOINTS)

def map_icd9_to_category(code: object) -> str:
    """
    Map ICD-9 codes into 9 disease chapters as represented in Strack et al. (2014).
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "missing"
    text = str(code).strip()
    if text in ("", "?", "nan"):
        return "missing"
    if text[0] in ("V", "E", "v", "e"):
        return "other"
    try:
        value = float(text)
    except ValueError:
        return "other"
    chapter = int(value) # Convert subchapter to chapter by truncating decimal places
    if 390 <= chapter <= 459 or chapter == 785:
        return "circulatory"
    if 460 <= chapter <= 519 or chapter == 786:
        return "respiratory"
    if 520 <= chapter <= 579 or chapter == 787:
        return "digestive"
    if chapter == 250:
        return "diabetes"
    if 800 <= chapter <= 999:
        return "injury"
    if 710 <= chapter <= 739:
        return "musculoskeletal"
    if 580 <= chapter <= 629 or chapter == 788:
        return "genitourinary"
    if 140 <= chapter <= 239:
        return "neoplasms"
    return "other"


def group_admission_type(admission_type_id: object) -> str:
    """
    Group admission types into 4 meaningful categories.
    IDs 5/6/8 are unknown. 7 (trauma center) is too rare to be its own category.
    """
    mapping = {1: "emergency", 2: "urgent", 3: "elective", 4: "newborn"}
    try:
        return mapping.get(int(admission_type_id), "other_or_unknown")
    except (ValueError, TypeError):
        return "other_or_unknown"


def group_discharge_disposition(discharge_disposition_id: object) -> str:
    """
    Group discharge 30 discharge-disposition IDs into 6 meaningful categories.
    """
    try:
        code = int(discharge_disposition_id)
    except (ValueError, TypeError):
        return "other_or_unknown"
    if code == 1:
        return "home"
    if code in (6, 8):
        return "home_with_home_health"
    if code in EXPIRED_OR_HOSPICE_DISPOSITIONS:
        return "expired_or_hospice"
    if code == 7:
        return "left_ama"
    if code in (2, 3, 4, 5, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30):
        return "transferred"
    return "other_or_unknown"


def group_admission_source(admission_source_id: object) -> str:
    """Group admission-source IDs into 4 meaningful categories."""
    try:
        code = int(admission_source_id)
    except (ValueError, TypeError):
        return "other_or_unknown"
    if code in (1, 2, 3):
        return "referral"
    if code == 7:
        return "emergency_room"
    if code in (4, 5, 6, 10, 18, 22, 25, 26):
        return "transfer"
    return "other_or_unknown"


def group_medical_specialty(specialty: object) -> str:
    """Group medical specialties into 7 meaningful categories."""
    if specialty is None or (isinstance(specialty, float) and np.isnan(specialty)):
        return "missing"
    text = str(specialty).strip().lower()
    if text in ("", "?", "nan"):
        return "missing"
    keep = {
        "InternalMedicine": "internal_medicine",
        "Emergency/Trauma": "emergency_trauma",
        "Family/GeneralPractice": "family_general_practice",
        "Cardiology": "cardiology",
    }
    if text in keep:
        return keep[text]
    if text.startswith("surgery") or text in ("surgery-general", "SurgicalSpecialty"):
        return "surgery"
    return "other"


# -----------------------------------------------------------------------------------------
# DataFrame-level transformations
# -----------------------------------------------------------------------------------------

def clean_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out encounters that are not relevant for predicting readmission. This includes:
    
    1. Convert '?' values to NaN.
    2. Drop encounters where the patient has died or is in hospice care.
    3. Keep only the first encounter for each patient, to preserve independence of samples and avoid data leakage.
    4. Drop the 'Unknown/Invalid' gender rows.
    """
    out = df.replace("?", np.nan)
    out = out[~out["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE_DISPOSITIONS)]
    out = out.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")
    out = out[out["gender"] != "Unknown/Invalid"]
    return out.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model feature matrix from cleaned encounter data.
    """
    out = df.copy()
    out = out.replace("?", np.nan)
    
    # Treat medications as 'No' if they are missing
    for col in MEDICATION_COLUMNS:
        if col not in out.columns:
            out[col] = "No"
        out[col] = out[col].fillna("No")
    
    # Engineered numeric features
    out["age_midpoint"] = out["age"].map(AGE_MIDPOINTS)
    out["service_utilization"] = (
        out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    )
    meds = out[MEDICATION_COLUMNS]
    out["n_medication_changes"] = meds.isin(["Up", "Down"]).sum(axis=1)
    out["n_active_medications"] = (meds != "No").sum(axis=1)
    
    # Grouped categorical features
    out["admission_type"] = out["admission_type_id"].map(group_admission_type)
    out["discharge_disposition"] = out["discharge_disposition_id"].map(group_discharge_disposition)
    out["admission_source"] = out["admission_source_id"].map(group_admission_source)
    out["medical_specialty_group"] = out.get("medical_specialty", pd.Series(np.nan, index=out.index)).map(group_medical_specialty)
        
    for diag_col in ("diag_1", "diag_2", "diag_3"):
        source = out.get(diag_col, pd.Series(np.nan, index=out.index))
        out[f"{diag_col}_category"] = source.map(map_icd9_to_category)
    
    out["race"] = out["race"].fillna("missing")
    # "None" here indicates the lab was not ordered
    out["max_glu_serum"] = out["max_glu_serum"].fillna("None")
    out["A1Cresult"] = out["A1Cresult"].fillna("None")
    
    return out[FEATURE_COLUMNS]
    

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build a preprocessing pipeline for the model features.
    Numeric features are imputed with the median and standardized.
    Categorical features are imputed with a constant value and one-hot encoded, ignoring rare categories.
    min_frequency=50 ensures ultra rare categories are placed into an "infrequent" bucket.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)),
    ])
    
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False
    )
    