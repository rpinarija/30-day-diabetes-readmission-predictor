# 30-Day Readmission Risk - Diabetes Cohort

Predicts unplanned 30-day hospital readmission risk for diabetic inpatients, so discharge-planning teams can prioritize follow-up for the patients most likely have an early readmission.

**Status: early / experimental.** EDA and the preprocessing pipeline are done and tested; model training, the FastAPI serving layer, and CI/deployment are not yet implemented (see [Repo structure](#repo-structure)). There are no results to report yet - no model has been trained.


## Quickstart

```bash
# 1. Install dependencies (requires Python 3.13, uv: https://docs.astral.sh/uv/)
uv sync --extra dev

# 2. Download the raw data (pulled from UCI's public archive)
uv run python scripts/download_data.py

# 3. Run the EDA notebook (documents every cleaning/feature decision)
uv run jupyter notebook notebooks/01_eda.ipynb
```

Training and inference are not runnable yet - `src/readmission/train.py` and `src/readmission/api/` are currently empty scaffolding. This section will be updated with `train` and `predict` commands once those land.

## Data

- **Source:** [Diabetes 130-US Hospitals, 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI ML Repository), described in Strack et al. (2014), *"Impact of HbA1c Measurement on Hospital Readmission Rates"*, BioMed Research International.
- **Obtaining it:** not committed to the repo. Run `scripts/download_data.py`, which fetches and unzips the official UCI archive into `data/raw/` (`diabetic_data.csv`, `IDS_mapping.csv`). No auth or VPN required since it's a public dataset.
- **Snapshot:** The UCI dataset has not been updated since 2014, so this is stable. 101,766 encounters across 130 hospitals.
- **PII/access:** Dataset is for public release by the original authors; no additional handling needed on our end.
- **Schema:** 50 raw columns (identifiers, demographics, admission metadata as coded integers, utilization counts, 3 ICD-9 diagnosis fields, 2 lab results, 23 medication columns, the `readmitted` label). Full column-by-column treatment is in the [EDA notebook](notebooks/01_eda.ipynb).
- **Split strategy:** one row per patient (first encounter only, to avoid leakage from repeat patients across train/test) after dropping death/hospice discharges, which can't be readmitted by definition. See the [EDA notebook §4](notebooks/01_eda.ipynb) for the leakage analysis behind this.

## Repo structure

```
├── data/
│   ├── raw/              # downloaded via scripts/download_data.py, gitignored
│   ├── processed/        # pipeline output, gitignored
├── notebooks/
│   ├── 01_eda.ipynb      # EDA + decision log - reasoning behid every decision
├── scripts/
│   ├── download_data.py  # Download UCI CSV
├── src/readmission/
│   ├── config.py         # paths, constants, risk-tier thresholds
│   ├── preprocessing.py  # cleaning, feature engineering, sklearn ColumnTransformer
│   ├── train.py          # empty - training pipeline not yet built
│   ├── api/              # empty - FastAPI serving layer not yet built
├── tests/                # scaffolded, not yet populated
```

## Approach

**Planned models** (training not yet implemented - `train.py` is still empty):

- **Class-weighted logistic regression** - interpretable baseline.
- **XGBoost** - the primary model. Handles the mix of skewed numeric counts and grouped categoricals without the linearity assumptions logistic regression needs, and is the standard robust choice for tabular data at this size.
- Both will use class weighting (not resampling) to keep predicted probabilities calibratable, per [decision #12 in the EDA log](notebooks/01_eda.ipynb).

**What's implemented so far:**

- **Cleaning & feature engineering** (`src/readmission/preprocessing.py`): drops non-predictive/high-missingness columns, collapses admission/discharge/source codes and ICD-9 diagnoses into clinically meaningful groups, engineers utilization and medication-activity counts, and builds an sklearn `ColumnTransformer` (median-impute + scale numerics, impute + one-hot categoricals).
- **Why these choices:** every decision (what to drop, how to group, how to handle informative missingness in lab results) is documented and justified in the [EDA notebook](notebooks/01_eda.ipynb), which also verifies the production pipeline reproduces the notebook's row count and target rate exactly.

The README will link to a training write-up here, and report which of the two models wins on the metrics below, once `train.py` exists.

## Evaluation

Not applicable yet - no model has been trained. The metric strategy is already decided (see [EDA notebook §13](notebooks/01_eda.ipynb) for full reasoning) because the ~9% positive rate rules out accuracy as meaningful:

- **ROC-AUC** - primary ranking metric, threshold-free and prevalence-independent.
- **PR-AUC** - reported alongside, since ROC-AUC can look optimistic under class imbalance.
- **Lift at top-k** - operational metric: how well the model concentrates risk in the top-ranked patients a care team would actually act on.
- **Brier score + ECE/reliability diagram** - calibration checks, since risk tiers (`src/readmission/config.py`) are only useful if a predicted 20% means an observed ~20%.

## Reproducibility

- Dependencies are pinned via `uv.lock`; `uv sync --extra dev` reproduces the exact environment.
- `RANDOM_STATE = 42` is centralized in `src/readmission/config.py` for reuse once train/test splitting is implemented.
- No experiment tracking or model artifact versioning exists yet - to be added alongside `train.py`.

## Deployment and operations

Not applicable - nothing is deployed. `src/readmission/api/` is scaffolded (`main.py`, `schemas.py`, `evaluation.py`) but empty. This section will be filled in once an inference endpoint exists.

## Housekeeping

- **Tests:** `uv run pytest` (test files exist under `tests/` but are currently empty placeholders).
- **Lint:** `uv run ruff check .`

## Citation

Strack, B., et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates:
Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International.