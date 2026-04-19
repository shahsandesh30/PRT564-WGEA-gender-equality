"""
visualisation.py — presentation-ready figures.

Owner: Pujan Dey (Visualisation & Reporting Lead)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIG_DIR
from .utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


def plot_regression_coefficients(model, feature_names: list[str],
                                 out: Path = FIG_DIR, top_n: int = 15) -> Path:
    """Signed coefficient bar chart for the OLS model (RQ1)."""
    coefs = pd.DataFrame({"feature": feature_names, "coef": model.coef_})
    coefs["abs"] = coefs["coef"].abs()
    coefs = coefs.sort_values("abs", ascending=False).head(top_n).sort_values("coef")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["firebrick" if c < 0 else "seagreen" for c in coefs["coef"]]
    ax.barh(coefs["feature"], coefs["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Linear regression coefficient")
    ax.set_title(f"RQ1 — OLS coefficients (top {top_n} by |coef|)")
    fig.tight_layout()
    path = out / "rq1_coefficients.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                             out: Path = FIG_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=15, color="navy")
    lims = [0, 1]
    ax.plot(lims, lims, color="red", linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual prop_women_mgmt")
    ax.set_ylabel("Predicted prop_women_mgmt")
    ax.set_title("RQ1 — Predicted vs Actual (test set)")
    ax.legend()
    fig.tight_layout()
    path = out / "rq1_pred_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path


def plot_confusion_matrix(cm: np.ndarray, out: Path = FIG_DIR,
                          labels: tuple[str, str] = ("Low", "High")) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("RQ2 — Confusion matrix (Naive Bayes)")
    fig.tight_layout()
    path = out / "rq2_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path
