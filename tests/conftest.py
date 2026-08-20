"""Shared test fixtures: a small synthetic raw-encounter DataFrame."""

import pandas as pd
import pytest

from readmission.preprocessing import MEDICATION_COLUMNS


def make_raw_encounter(**overides) -> dict:
    """One raw encounter record, with default values for all fields; override any field."""
    row = {
        "encounter_id": 1,
        "patient_nbr": 100,
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 1,
        "time_in_hospital": 5,
        "payer_code": "?",
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
        "diag_1": "428",
        "diag_2": "215.02",
        "diag_3": "401",
        "number_diagnoses": 5,
        "max_glu_serum": "None",
        "A1Cresult": ">8",
        "change": "Ch",
        "diabetesMed": "Yes",
        "readmitted": "NO",
    }
    for med in MEDICATION_COLUMNS:
        row[med] = "No"
    row["insulin"] = "Up"
    row["metformin"] = "Steady"
    row.update(overides)
    return row

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """"Five encounters covering the cleaning edge cases:
    - rows 1-2: same patient twice (dedup should keep the earlier encounter)
    - row 3: discharged to hospice (should be dropped)
    - row 4: gender Uknown/Invalid (should be dropped)
    - row 5: an ordinary readmitted within 30-days encounter (should be kept)
    """
    rows = [
        make_raw_encounter(encounter_id=10, patient_nbr=1, readmitted="NO"),
        make_raw_encounter(encounter_id=11, patient_nbr=1, readmitted="<30"),
        make_raw_encounter(encounter_id=12, patient_nbr=2, discharge_disposition_id=13),
        make_raw_encounter(encounter_id=13, patient_nbr=3, gender="Unknown/Invalid"),
        make_raw_encounter(encounter_id=14, patient_nbr=4, readmitted="<30", race="?"),
    ]
    return pd.DataFrame(rows)