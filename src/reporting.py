"""
reporting.py — collect metrics and t-test results into tables.

Owner: Pujan Dey (Visualisation & Reporting Lead)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import TABLE_DIR
from .utils import get_logger, save_table

logger = get_logger(__name__)


def write_metrics_summary(
    reg_metrics: dict[str, float],
    cls_metrics_rf: dict[str, float],
    cls_metrics_nb: dict[str, float],
    out: Path = TABLE_DIR,
) -> Path:
    rows = [
        {"model": "OLS (RQ1)", **reg_metrics},
        {"model": "RandomForest (RQ2)", **cls_metrics_rf},
        {"model": "NaiveBayes (RQ2 baseline)", **cls_metrics_nb},
    ]
    df = pd.DataFrame(rows)
    path = out / "metrics_summary.csv"
    save_table(df, path)
    logger.info("Metrics summary → %s", path)
    return path


def write_ttest_results(ttest_reg: dict, ttest_cls: dict, out: Path = TABLE_DIR) -> Path:
    path = out / "paired_ttest_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rq1_ols_vs_ridge": ttest_reg, "rq2_rf_vs_nb": ttest_cls}, f, indent=2)
    logger.info("Paired t-test results → %s", path)
    return path


def write_feature_importance(importances: pd.DataFrame, out: Path = TABLE_DIR) -> Path:
    path = out / "rq2_feature_importance.csv"
    save_table(importances, path)
    return path


def write_vif(vif_df: pd.DataFrame, out: Path = TABLE_DIR) -> Path:
    path = out / "rq1_vif.csv"
    save_table(vif_df, path)
    return path
