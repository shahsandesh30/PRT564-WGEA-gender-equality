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
    reg_metrics_simple: dict[str, float],
    reg_metrics_multi: dict[str, float],
    cls_metrics_nb: dict[str, float],
    out: Path = TABLE_DIR,
) -> Path:
    """Write a single CSV summarising all model metrics for the slide deck."""
    rows = [
        {"model": "Simple LR (RQ1 baseline)", **reg_metrics_simple},
        {"model": "Multiple LR (RQ1 full)", **reg_metrics_multi},
        {"model": "Naive Bayes (RQ2)", **cls_metrics_nb},
    ]
    df = pd.DataFrame(rows)
    path = out / "metrics_summary.csv"
    save_table(df, path)
    logger.info("Metrics summary → %s", path)
    return path


def write_ttest_results(ttest_reg: dict, ttest_cls: dict, out: Path = TABLE_DIR) -> Path:
    """Persist the two paired-CV t-test results (regression and classification)."""
    path = out / "paired_ttest_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "rq1_simple_vs_multiple_lr": ttest_reg,
                "rq2_naive_bayes_vs_baseline": ttest_cls,
            },
            f,
            indent=2,
        )
    logger.info("Paired t-test results → %s", path)
    return path


def write_vif(vif_df: pd.DataFrame, out: Path = TABLE_DIR) -> Path:
    path = out / "rq1_vif.csv"
    save_table(vif_df, path)
    return path
