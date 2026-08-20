"""Unit tests for readmission.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from readmission.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessing_pipeline,
    clean_encounters,
    engineer_features,
    group_admission_source,
    group_admission_type,
    group_discharge_disposition,
    group_medical_specialty,
    make_binary_target,
    map_icd9_to_category,
)
from tests.conftest import make_raw_encounter


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------
class TestMakeBinaryTarget:
    def test_only_under_30_is_positive(self):
        labels = pd.Series(["<30", ">30", "NO"])
        assert make_binary_target(labels).tolist() == [1, 0, 0]
        
    def test_returns_int_dtype(self):
        assert make_binary_target(pd.Series(["No"])).dtype == np.int64

  
# ---------------------------------------------------------------------------
# ICD-9 grouping
# ---------------------------------------------------------------------------
class TestMapIcd9ToCategory:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("428", "circulatory"),
            ("785.4", "circulatory"),
            ("486", "respiratory"),
            ("786.05", "respiratory"),
            ("531", "digestive"),
            ("250.83", "diabetes"),
            ("250", "diabetes"),
            ("805", "injury"),
            ("999", "injury"),
            ("715", "musculoskeletal"),
            ("585", "genitourinary"),
            ("788.20", "genitourinary"),
            ("162", "neoplasms"),
            ("V57", "other"),
            ("E909", "other"),
            ("780", "other"),
            (8, "other"),
        ],
    )
    def test_known_mappings(self, code, expected):
        assert map_icd9_to_category(code) == expected
    
    @pytest.mark.parametrize("code", [None, np.nan, "?", "nan"])
    def test_missing_values(self, code):
        assert map_icd9_to_category(code) == "missing"
    
    def test_garbage_values(self):
        assert map_icd9_to_category("not-a-code") == "other"


# ---------------------------------------------------------------------------
# ID-code grouping
# ---------------------------------------------------------------------------

class TestGroupAdmissionType:
    def test_admission_type(self):
        assert group_admission_type(1) == "emergency"
        assert group_admission_type(3) == "elective"
        assert group_admission_type(5) == "other_or_unknown"
        assert group_admission_type("2") == "urgent"
        assert group_admission_type(None) == "other_or_unknown"
        
    def test_discharge_disposition(self):
        assert group_discharge_disposition(1) == "home"
        assert group_discharge_disposition(6) == "home_with_home_health"
        assert group_discharge_disposition(3) == "transferred"
        assert group_discharge_disposition(7) == "left_ama"
        assert group_discharge_disposition(11) == "expired_or_hospice"
        assert group_discharge_disposition(25) == "other_or_unknown"
    
    def test_admission_source(self):
        assert group_admission_source(7) == "emergency_room"
        assert group_admission_source(1) == "referral"
        assert group_admission_source(4) == "transfer"
        assert group_admission_source(9) == "other_or_unknown"
        
    def test_medical_specialty(self):
        assert group_medical_specialty("InternalMedicine") == "internal_medicine"
        assert group_medical_specialty("Surgery-Cardiovascular/Thoracic") == "surgery"
        assert group_medical_specialty("Pediatrics-Endocrinology") == "other"
        assert group_medical_specialty(np.nan) == "missing"
        assert group_medical_specialty("?") == "missing"
        

# ---------------------------------------------------------------------------
# Encounter-level cleaning
# ---------------------------------------------------------------------------

class TestCleanEncounters:
    def test_remove_expired_or_hospice(self, raw_df):
        cleaned = clean_encounters(raw_df)
        assert 12 not in cleaned["encounter_id"].values
    
    def test_keep_first_encounter_per_patient(self, raw_df):
        cleaned = clean_encounters(raw_df)
        patient_rows = cleaned[cleaned["patient_nbr"] == 1]
        assert len(patient_rows) == 1
        assert patient_rows["encounter_id"].iloc[0] == 10
        
    def test_drops_unknown_gender(self, raw_df):
        cleaned = clean_encounters(raw_df)
        assert "Unknown/Invalid" not in cleaned["gender"].values
        
    def test_questiion_marks_become_nan(self, raw_df):
        cleaned = clean_encounters(raw_df)
        row = cleaned[cleaned["patient_nbr"] == 4]
        assert row["race"].isna().all()
        
    def test_expected_row_count(self, raw_df):
        # 5 rows - 1 (duplicate) - 1 (hospice) - 1 (unknown gender) = 2 rows
        assert len(clean_encounters(raw_df)) == 2
        

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_output_columns(self, raw_df):
        features = engineer_features(clean_encounters(raw_df))
        assert list(features.columns) == FEATURE_COLUMNS
    
    def test_age_midpoint(self):
        df = pd.DataFrame([make_raw_encounter(age="[70-80)")])
        assert engineer_features(df)["age_midpoint"].iloc[0] == 75
        
    def test_service_utilization_is_sum_of_prior_visits(self):
        df = pd.DataFrame([make_raw_encounter(number_outpatient=2, number_emergency=3, number_inpatient=4)])
        assert engineer_features(df)["service_utilization"].iloc[0] == 9
    
    def test_medication_counts(self):
        # insulin=Up counts as changed AND active; metformin=Steady only active
        df = pd.DataFrame([make_raw_encounter(insulin="Up", metformin="Steady")])
        features = engineer_features(df)
        assert features["n_medication_changes"].iloc[0] == 1
        assert features["n_active_medications"].iloc[0] == 2
    
    def test_missing_medication_columns_default_to_no(self):
        row = make_raw_encounter()
        for med in ("examide", "citoglipton", "troglitazone"):
            row.pop(med)
        features = engineer_features(pd.DataFrame([row]))
        assert features["n_active_medications"].iloc[0] == 2 #insulin + metformin

    def test_diagnosis_categories(self):
        df = pd.DataFrame([make_raw_encounter(diag_1="428", diag_2="250.83", diag_3="V45")])
        features = engineer_features(df)
        assert features["diag_1_category"].iloc[0] == "circulatory"
        assert features["diag_2_category"].iloc[0] == "diabetes"
        assert features["diag_3_category"].iloc[0] == "other"
    
    def test_single_row_serving_path(self):
        """The API scores on row at a time - must not require a batch of rows to work."""
        features = engineer_features(pd.DataFrame([make_raw_encounter()]))
        assert len(features) == 1
        assert not features[NUMERIC_FEATURES].isna().any().any()
    
    def test_unseen_missing_race_becomes_category(self):
        df = pd.DataFrame([make_raw_encounter(race="?")])
        assert engineer_features(df)["race"].iloc[0] == "missing"
 
        
# ---------------------------------------------------------------------------
# Sklearn preprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def test_fit_transform_produces_finite_matrix(self, raw_df):
        features = engineer_features(clean_encounters(raw_df))
        preprocessor = build_preprocessing_pipeline()
        matrix = preprocessor.fit_transform(features)
        assert matrix.shape[0] == len(features)
        assert np.isfinite(matrix).all()
    
    def test_unknown_category_at_serving_time_does_ot_crash(self, raw_df):
        features = engineer_features(clean_encounters(raw_df))
        preprocessor = build_preprocessing_pipeline()
        preprocessor.fit(features)
        novel = features.copy()
        novel.loc[novel.index[0], "race"] = "Martian"
        matrix = preprocessor.transform(novel)
        assert np.isfinite(matrix).all()
    
    def test_numeric_and_categorical_lists_are_disjoint(self):
        assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)