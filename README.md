# PRT564 Assessment 2 — Analysing Gender Equality in Australian Workplaces

**Group 9 — Charles Darwin University, Semester 1 2026**
Dataset: WGEA Public Data File 2025 (https://data.gov.au/data/dataset/wgea-dataset)

> **Data folders** (`data/` and `Dataset/`) are excluded from this repo due to file size.
> Download them from Google Drive: https://drive.google.com/file/d/1tG60DqyY-izVwYdzRwekNvRrJIxWn4lU/view?usp=sharing

This repo implements the analysis pipeline planned in Assessment 1: multiple
linear regression, classification, EDA, and statistical significance testing
on the WGEA 2025 employer-level data.

## Team & module ownership

| Member            | Role                                  | Owns                                                            |
|-------------------|---------------------------------------|-----------------------------------------------------------------|
| Shuvechchha Pun   | Data Acquisition & Context Lead       | `src/config.py`, `src/data_acquisition.py`                      |
| Aadarsh Ghimire   | Preprocessing & Research Design Lead  | `src/preprocessing.py`, `src/feature_engineering.py`            |
| Sandesh Shahi     | Analysis Lead                         | `src/regression.py`, `src/classification.py`, `src/evaluation.py` |
| Pujan Dey         | Visualisation & Reporting Lead        | `src/eda.py`, `src/visualisation.py`, `src/reporting.py`, `main.py` |

## Research questions (from Assessment 1)

- **RQ1** — Multiple linear regression: predict proportion of women in management from industry, size, and policy adoption.
- **RQ2** — Classification: High vs Low women-in-management based on policy + structural characteristics (Random Forest vs Naive Bayes baseline).
- **RQ3** — Descriptive: gender composition across ANZSIC divisions and employer-size bands.
- **RQ4** — Diagnostic: correlation between parental-leave / flexible-work policies and women-in-workforce.

## Folder layout

```
Assessment2/
├── Dataset/                                 # raw WGEA CSVs (and 5-row samples)
├── data/
│   ├── external/                            # heterogeneous source (ABS pay-gap CSV)
│   └── processed/employer_level.parquet     # merged employer-level master
├── src/                                     # analysis modules (see ownership above)
├── outputs/
│   ├── figures/                             # PNGs for slides
│   ├── tables/                              # metric CSVs + t-test JSON
│   └── models/
├── main.py                                  # end-to-end orchestrator
└── requirements.txt
```

## Running — recommended: step-by-step notebooks

The primary workflow for learning, reviewing, and building the presentation is the **staged notebook pipeline** in `notebooks/`. Each notebook runs one stage, shows intermediate DataFrames and plots inline, and writes a checkpoint that the next notebook loads. Run them in order:

| # | Notebook                              | Produces (in `data/processed/checkpoints/`)                                |
|---|---------------------------------------|----------------------------------------------------------------------------|
| 01 | `01_data_acquisition.ipynb`          | `01_raw_data.pkl` (dict of 8 DataFrames + optional ABS)                    |
| 02 | `02_preprocessing.ipynb`             | `02_master.parquet` (one row per employer)                                 |
| 03 | `03_feature_engineering.ipynb`       | `03_master_with_targets.parquet`, `03_X.parquet`, `03_y_reg.parquet`, `03_y_cls.parquet`, `03_feature_names.json` |
| 04 | `04_eda.ipynb`                       | Figures in `outputs/figures/`, `outputs/tables/rq1_vif.csv`                |
| 05 | `05_regression.ipynb`                | `05_ols_model.pkl`, `05_reg_metrics.json`                                  |
| 06 | `06_classification.ipynb`            | `06_rf_model.pkl`, `06_nb_model.pkl`, `06_cls_metrics.json`                |
| 07 | `07_statistical_tests.ipynb`         | `07_ttest_results.json` (paired-CV t-tests for both RQs)                   |
| 08 | `08_reporting.ipynb`                 | `outputs/tables/metrics_summary.csv`, narrative template for slides         |

```bash
pip install -r requirements.txt
jupyter notebook notebooks/       # run cells top-to-bottom, 01 → 08
```

Each notebook is self-contained: if you only change step 05, you only need to re-run 05 (and downstream if they depend on its checkpoint). Notebooks can also be re-executed individually during presentation rehearsals to show the examiner live.

## Running — batch mode (`main.py`)

For a single end-to-end run without opening notebooks:

```bash
# Fast wiring test on the 5-row sample files
USE_SAMPLE=1 python main.py

# Full run
python main.py
```

Artefacts the presentation uses:
- `outputs/figures/rq3_women_share_by_division.png`
- `outputs/figures/rq3_women_share_by_size.png`
- `outputs/figures/rq4_policy_vs_women.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/rq1_coefficients.png`
- `outputs/figures/rq1_pred_vs_actual.png`
- `outputs/figures/rq1_residuals_vs_fitted.png` + `rq1_qq_plot.png` (assumption checks)
- `outputs/figures/rq2_feature_importance.png`
- `outputs/figures/rq2_confusion_matrix.png`
- `outputs/tables/metrics_summary.csv`
- `outputs/tables/paired_ttest_results.json`
- `outputs/tables/rq1_vif.csv`

## Heterogeneous data integration

To integrate the secondary source required for the higher grade band, drop a
CSV at `data/external/abs_gender_pay_gap_by_industry.csv` with at least these
columns:

| anzsic_division                               | industry_pay_gap |
|-----------------------------------------------|------------------|
| Financial and Insurance Services              | 22.7             |
| Health Care and Social Assistance             | 17.4             |
| ...                                           | ...              |

The pipeline auto-detects and left-joins it on `anzsic_division`. If the file
is absent, the pipeline still runs on WGEA data alone.

## Rubric → module map

| Assessment 2 rubric component                    | Where it lives                                        |
|--------------------------------------------------|-------------------------------------------------------|
| 1. Preprocessing + heterogeneous integration     | `src/preprocessing.py`                                |
| 2. EDA                                           | `src/eda.py`                                          |
| 3. Regression                                    | `src/regression.py`                                   |
| 4. Model evaluation + statistical tests          | `src/evaluation.py` (incl. `paired_t_test_cv`)        |
| 5. Presentation artefacts                        | `outputs/figures/`, `outputs/tables/`                 |
