"""
regression.py — RQ1 Linear Regression (simple + multiple).

Owner: Sandesh Shahi (Analysis Lead)

Implements Assessment 1 Section 4.2. A single fit helper is used for both
the simple baseline (one predictor) and the full multiple-regression model
(all predictors). Only scikit-learn's LinearRegression (OLS) is used — the
method taught in Weeks 1 and 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .config import FIG_DIR, RANDOM_SEED, TEST_SIZE
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class RegressionResult:
    model: LinearRegression
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred_test: np.ndarray


def fit_linear_regression(X: pd.DataFrame, y: pd.Series) -> RegressionResult:
    """Fit OLS with an 80/20 split and the project random seed.

    Works for both the Simple Linear Regression baseline (single column X)
    and the Multiple Linear Regression full model (many-column X).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.info("OLS fitted: %d train / %d test samples, %d features",
                len(X_train), len(X_test), X.shape[1])
    return RegressionResult(model, X_train, X_test, y_train, y_test, y_pred)


def diagnostics(res: RegressionResult, out: Path = FIG_DIR,
                prefix: str = "rq1") -> dict[str, Path]:
    """
    Produce residual-vs-fitted and Q-Q plots.

    These visualise the linear-regression assumptions declared in
    Assessment 1 Section 4.2: linearity, homoscedasticity, and normality.
    """
    residuals = res.y_test.values - res.y_pred_test
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(res.y_pred_test, residuals, alpha=0.35, s=15, color="navy")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values (predicted prop_women_mgmt)")
    ax.set_ylabel("Residuals")
    ax.set_title("OLS — Residuals vs Fitted (homoscedasticity check)")
    fig.tight_layout()
    p = out / f"{prefix}_residuals_vs_fitted.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["residuals_vs_fitted"] = p

    fig, ax = plt.subplots(figsize=(6, 4.5))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("OLS — Q-Q plot (normality of residuals)")
    fig.tight_layout()
    p = out / f"{prefix}_qq_plot.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["qq_plot"] = p

    logger.info("Saved regression diagnostic plots.")
    return paths
