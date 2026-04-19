# PRT564 Assessment 2 — Analysing Gender Equality in Australian Workplaces

**Group 9 — Charles Darwin University, Semester 1 2026**
Dataset: WGEA Public Data File 2025 (https://data.gov.au/data/dataset/wgea-dataset)

> **Data folders** (`data/` and `Dataset/`) are excluded from this repo due to file size.
> Download them from Google Drive: https://drive.google.com/file/d/1tG60DqyY-izVwYdzRwekNvRrJIxWn4lU/view?usp=sharing

This repo implements the analysis pipeline planned in Assessment 1: linear regression (simple + multiple), classification (Naive Bayes), EDA, and statistical significance testing on the WGEA 2025 employer-level data. 

## Team & module ownership

| Member            | Role                                  | Owns                                                            |
|-------------------|---------------------------------------|-----------------------------------------------------------------|
| Shuvechchha Pun   | Data Acquisition & Context Lead       | `src/config.py`, `src/data_acquisition.py`                      |
| Aadarsh Ghimire   | Preprocessing & Research Design Lead  | `src/preprocessing.py`, `src/feature_engineering.py`            |
| Sandesh Shahi     | Analysis Lead                         | `src/regression.py`, `src/classification.py`, `src/evaluation.py` |
| Pujan Dey         | Visualisation & Reporting Lead        | `src/eda.py`, `src/visualisation.py`, `src/reporting.py`        |

## Research questions (from Assessment 1)

- **RQ1** — Linear regression: predict proportion of women in management from industry, size, and policy adoption. Simple LR (baseline) vs Multiple LR (full model).
- **RQ2** — Classification: High vs Low women-in-management using Gaussian Naive Bayes, compared against a majority-class baseline.
- **RQ3** — Descriptive: gender composition across ANZSIC divisions and employer-size bands.
- **RQ4** — Diagnostic: association between parental-leave / flexible-work policies and women-in-workforce.

## Folder layout

```
project/
├── Dataset/                                 # raw WGEA CSVs
├── data/
│   └── processed/checkpoints/               # notebook-to-notebook handoffs
├── notebooks/                               # 01 → 08 staged pipeline
├── src/                                     # analysis modules 
├── outputs/
│   ├── figures/                             # PNGs for slides
│   └── tables/                              # metric CSVs + t-test JSON
└── requirements.txt
```

## Running the pipeline

```bash
pip install -r requirements.txt
jupyter notebook notebooks/       # run cells top-to-bottom, 01 → 08
```

Each notebook runs one stage, shows intermediate DataFrames and plots inline, and writes a checkpoint that the next notebook loads:

| # | Notebook                              | Produces                                                                    |
|---|---------------------------------------|-----------------------------------------------------------------------------|
| 01 | `01_data_acquisition.ipynb`          | `01_raw_data.pkl` — dict of 8 DataFrames + optional ABS                     |
| 02 | `02_preprocessing.ipynb`             | `02_master.parquet` — one row per employer                                  |
| 03 | `03_feature_engineering.ipynb`       | `03_X.parquet`, `03_y_reg.parquet`, `03_y_cls.parquet`, `03_feature_names.json`, `03_master_with_targets.parquet` |
| 04 | `04_eda.ipynb`                       | EDA figures in `outputs/figures/`, `outputs/tables/rq1_vif.csv`             |
| 05 | `05_regression.ipynb`                | `05_ols_simple.pkl`, `05_ols_multiple.pkl`, `05_reg_metrics.json`           |
| 06 | `06_classification.ipynb`            | `06_nb_model.pkl`, `06_cls_metrics.json`                                    |
| 07 | `07_statistical_tests.ipynb`         | `07_ttest_results.json` (paired-CV t-tests for both RQs)                    |
| 08 | `08_reporting.ipynb`                 | `outputs/tables/metrics_summary.csv`, slide-deck narrative template         |

Each notebook is self-contained: if you only change step 05, you only need to re-run 05 (and any downstream notebooks that depend on its checkpoint).

## Artefacts the presentation uses

- `outputs/figures/rq3_women_share_by_division.png`, `rq3_women_share_by_size.png` — RQ3
- `outputs/figures/rq4_policy_vs_women.png`, `correlation_heatmap.png` — RQ4
- `outputs/figures/rq1_coefficients.png`, `rq1_pred_vs_actual.png` — RQ1 model
- `outputs/figures/rq1_residuals_vs_fitted.png`, `rq1_qq_plot.png` — RQ1 assumption checks
- `outputs/figures/rq2_confusion_matrix.png` — RQ2 classifier
- `outputs/tables/metrics_summary.csv` — headline numbers
- `outputs/tables/paired_ttest_results.json` — hypothesis-test results
- `outputs/tables/rq1_vif.csv` — multicollinearity check
