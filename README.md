# 30-Day Readmission Risk - Diabetes Cohort

Predicts unplanned 30-day hospital readmission risk for diabetic inpatients, so discharge-planning teams can prioritize follow-up for the patients most likely to have an early readmission.

**Status: in progress.** EDA, the preprocessing pipeline, and the training/evaluation pipeline are done and tested. The FastAPI serving layer and CI/deployment are not yet implemented (see [Repo structure](#repo-structure)).


## Quickstart

```bash
# 1. Install dependencies (requires Python 3.13, uv: https://docs.astral.sh/uv/)
uv sync --extra dev

# 2. Download the raw data (pulled from UCI's public archive)
uv run python scripts/download_data.py

# 3. Train, calibrate, and evaluate the model (writes artifacts to models/)
uv run python -m readmission.train

# 4. (Optional) Run the EDA notebook (documents every cleaning/feature decision)
uv run --with jupyter jupyter notebook notebooks/01_eda.ipynb
```

Inference is not runnable yet - `src/readmission/api/` is still empty scaffolding. This section will be updated with a `predict`/serve command once that lands.

## Data

- **Source:** [Diabetes 130-US Hospitals, 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI ML Repository), described in Strack et al. (2014), *"Impact of HbA1c Measurement on Hospital Readmission Rates"*, BioMed Research International.
- **Obtaining it:** not committed to the repo. Run `scripts/download_data.py`, which fetches and unzips the official UCI archive into `data/raw/` (`diabetic_data.csv`, `IDS_mapping.csv`). No auth or VPN required since it's a public dataset.
- **Snapshot:** The UCI dataset has not been updated since 2014, so this is stable. 101,766 encounters across 130 hospitals.
- **PII/access:** Dataset is for public release by the original authors; no additional handling needed on our end.
- **Schema:** 50 raw columns (identifiers, demographics, admission metadata as coded integers, utilization counts, 3 ICD-9 diagnosis fields, 2 lab results, 23 medication columns, the `readmitted` label). Full column-by-column treatment is in the [EDA notebook](notebooks/01_eda.ipynb).
- **Cohort definition:** one row per patient (first encounter only, to avoid leakage from repeat patients across train/test) after dropping death/hospice discharges, which can't be readmitted by definition. See the [EDA notebook §4](notebooks/01_eda.ipynb) for the leakage analysis behind this. This leaves 69,987 patients with a ~9.0% positive rate.
- **Splits:** 80/20 stratified train/test split. Within the training set, a further 80/20 stratified split separates the data used to fit the final model from a held-out calibration set, so calibration never sees the data the model was fit on and the test set is touched only once, for final scoring.

## Repo structure

```
├── data/
│   ├── raw/              # downloaded via scripts/download_data.py, gitignored
│   ├── processed/        # pipeline output, gitignored
├── models/               # training outputs (model, metrics, plots), gitignored
├── notebooks/
│   ├── 01_eda.ipynb      # EDA + decision log - reasoning behind every decision
├── scripts/
│   ├── download_data.py  # Download UCI CSV
├── src/readmission/
│   ├── config.py         # paths, constants, risk-tier thresholds
│   ├── preprocessing.py  # cleaning, feature engineering, sklearn ColumnTransformer
│   ├── train.py          # model selection, tuning, calibration, artifact saving
│   ├── evaluation.py     # metrics, baselines, lift, calibration diagnostics
│   ├── api/              # empty - FastAPI serving layer not yet built
├── tests/                # pytest suite for preprocessing and evaluation (API tests pending)
```

## Approach

**Models compared:**

- **Class-weighted logistic regression** - interpretable baseline.
- **XGBoost** - handles the mix of skewed numeric counts and grouped categoricals without the linearity assumptions logistic regression needs, and is the standard robust choice for tabular data at this size.
- Both use class weighting (not resampling) to keep predicted probabilities calibratable, per [decision #12 in the EDA log](notebooks/01_eda.ipynb).

**Training pipeline** (`src/readmission/train.py`):

1. Both candidates are compared with 5-fold stratified cross-validation on the training set. Preprocessing lives inside each sklearn `Pipeline`, so imputation/scaling/encoding is re-fit per fold with no leakage.
2. The family with the higher mean CV ROC-AUC is tuned with a 30-trial randomized hyperparameter search (also 5-fold CV, training set only).
3. The tuned model is refit on the fit split and calibrated with isotonic regression on the disjoint calibration split.
4. Every headline score is reported next to two reference points: a no-skill baseline and a no-ML incumbent rule (rank patients by prior inpatient visits).

**Cleaning & feature engineering** (`src/readmission/preprocessing.py`): drops non-predictive/high-missingness columns, collapses admission/discharge/source codes and ICD-9 diagnoses into clinically meaningful groups, engineers utilization and medication-activity counts, and builds an sklearn `ColumnTransformer` (median-impute + scale numerics, impute + one-hot categoricals). Every decision is documented and justified in the [EDA notebook](notebooks/01_eda.ipynb), which also verifies the production pipeline reproduces the notebook's row count and target rate exactly.

## Evaluation

The ~9% positive rate rules out accuracy as meaningful (predicting "no readmission" for everyone scores 91%), so the metric strategy is (see [EDA notebook §13](notebooks/01_eda.ipynb) for full reasoning):

- **ROC-AUC** - primary ranking metric, threshold-free and prevalence-independent.
- **PR-AUC** - reported alongside, since ROC-AUC can look optimistic under class imbalance.
- **Lift at top-k** - operational metric: how well the model concentrates risk in the top-ranked patients a care team would actually act on.
- **Brier score + ECE/reliability diagram** - calibration checks, since risk tiers (`src/readmission/config.py`) are only useful if a predicted 20% means an observed ~20%.

**Results** (held-out test set, n = 13,998; XGBoost won model selection):

| Metric | No-skill | Incumbent (`number_inpatient`) | Model (calibrated) |
|---|---|---|---|
| ROC-AUC | 0.500 | 0.552 | **0.651** |
| PR-AUC | 0.090 | 0.116 | **0.159** |
| Brier | 0.0817 | - | **0.0795** |
| Lift @ top 10% | 1.00x | 1.81x | **2.32x** |

- **Ranking:** the top 10% of patients by predicted risk have a 20.9% readmission rate (2.3x the base rate), and the top 5% have 23.6% (2.6x).
- **Calibration:** isotonic calibration cut expected calibration error from 0.372 to 0.006 with almost no change in ROC-AUC, so the predicted probabilities can be read as real risks.
- **Risk tiers:** observed readmission rates are 6.2% (low, 62% of patients), 12.5% (moderate, 34%), and 25.5% (high, 3%).
- **Honest read:** an ROC-AUC of 0.65 is modest. This is consistent with published work on this dataset, which has no post-discharge, social, or vitals data. The model is useful for prioritizing a follow-up list, not as a standalone decision rule.

Full metrics are written to `models/metrics.json`, and the reliability diagram to `models/reliability.png`.

## Reproducibility

- Dependencies are pinned via `uv.lock`; `uv sync --extra dev` reproduces the exact environment.
- `RANDOM_STATE = 42` is centralized in `src/readmission/config.py` and used for every split, CV fold, search, and model seed.
- Each training run saves the calibrated model with its version and timestamp (`models/model.joblib`), all metrics (`models/metrics.json`), and appends every tuning trial to `models/tuning_runs.jsonl`. There is no external experiment tracker yet.

## Deployment and operations

Not applicable - nothing is deployed. `src/readmission/api/` is scaffolded (`main.py`, `schemas.py`) but empty. This section will be filled in once an inference endpoint exists.

## Housekeeping

- **Tests:** `uv run pytest` (covers preprocessing and evaluation; `tests/test_api.py` is an empty placeholder until the API exists).
- **Lint:** `uv run ruff check .`

## Citation

Strack, B., et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates:
Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International.
