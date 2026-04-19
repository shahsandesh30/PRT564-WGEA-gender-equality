"""
evaluation.py — metrics, assumption diagnostics, and statistical tests.

Owner: Sandesh Shahi (Analysis Lead)

Metric choices tie directly to Assessment 1 Sections 4.2/4.3 and justify the
statistical rationale required by Assessment 2 rubric item 4:

  * MAE  — robust to outliers, interpretable on the same scale as y.
  * RMSE — penalises large errors; relevant because management proportions
           have a bounded [0, 1] range so big misses are important.
  * R² / Adjusted R² — share of variance explained; Adjusted R² corrects
           for the number of predictors, preventing spurious inflation.
  * Precision/Recall/F1 — appropriate for the RQ2 binary classification
           (balanced after median split but reported for completeness).
  * Paired-CV t-test (scipy.stats.ttest_rel) — compares per-fold scores of
           two competing models to test whether one outperforms the other
           beyond random CV variation. Hypotheses:
             H0: mean difference in scores = 0 (no real difference).
             H1: mean difference ≠ 0 (one model is statistically better).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from .config import CV_FOLDS, RANDOM_SEED
from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    # Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Adj_R2": adj_r2}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_proba: np.ndarray | None = None) -> dict[str, float]:
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            m["ROC_AUC"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            m["ROC_AUC"] = float("nan")
    return m


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


# ---------------------------------------------------------------------------
# Paired-CV t-test
# ---------------------------------------------------------------------------
def paired_t_test_cv(
    model_a,
    model_b,
    X: pd.DataFrame,
    y: pd.Series,
    scoring_fn: Callable[[np.ndarray, np.ndarray], float],
    stratify: bool = False,
    higher_is_better: bool = True,
    name_a: str = "A",
    name_b: str = "B",
    X_b: pd.DataFrame | None = None,
) -> dict:
    """
    Run k-fold CV for two models on identical folds, then scipy.stats.ttest_rel.

    scoring_fn(y_true, y_pred) -> float. For regression pass e.g. mean_absolute_error
    (with higher_is_better=False) or r2_score; for classification pass accuracy_score.

    If X_b is supplied, model_b is trained/tested on that feature matrix instead of X.
    Useful when comparing a simple baseline (fewer features) against a full model on the
    same row indices and same folds.
    """
    cv = (StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
          if stratify else
          KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED))

    X_b = X if X_b is None else X_b
    if len(X_b) != len(X):
        raise ValueError("X and X_b must share the same row index length for paired testing.")

    scores_a, scores_b = [], []
    y_arr = np.asarray(y)
    for train_idx, test_idx in cv.split(X, y_arr):
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        ma = clone(model_a).fit(X.iloc[train_idx], y_tr)
        mb = clone(model_b).fit(X_b.iloc[train_idx], y_tr)
        scores_a.append(scoring_fn(y_te, ma.predict(X.iloc[test_idx])))
        scores_b.append(scoring_fn(y_te, mb.predict(X_b.iloc[test_idx])))

    scores_a, scores_b = np.array(scores_a), np.array(scores_b)
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)

    better = name_a if (scores_a.mean() > scores_b.mean()) == higher_is_better else name_b
    result = {
        "model_a": name_a,
        "model_b": name_b,
        "mean_a": float(scores_a.mean()),
        "mean_b": float(scores_b.mean()),
        "std_a": float(scores_a.std(ddof=1)),
        "std_b": float(scores_b.std(ddof=1)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_at_0.05": bool(p_val < 0.05),
        "better_model": better,
        "scores_a": scores_a.tolist(),
        "scores_b": scores_b.tolist(),
    }
    logger.info(
        "Paired t-test %s vs %s: mean_a=%.4f mean_b=%.4f t=%.3f p=%.4f",
        name_a, name_b, result["mean_a"], result["mean_b"],
        result["t_statistic"], result["p_value"],
    )
    return result
