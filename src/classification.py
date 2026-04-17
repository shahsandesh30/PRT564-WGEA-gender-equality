"""
classification.py — RQ2 Random Forest + Naive Bayes baseline.

Owner: Sandesh Shahi (Analysis Lead)

Implements Assessment 1 Section 4.3. The RF vs NB comparison is fed into
evaluation.paired_t_test_cv to satisfy the statistical-significance
requirement in Assessment 2 rubric item 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from .config import RANDOM_SEED, TEST_SIZE
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    model: object
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred_test: np.ndarray
    y_proba_test: np.ndarray | None


def _split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=y if y.nunique() > 1 else None,
    )


def fit_random_forest(X: pd.DataFrame, y: pd.Series) -> ClassificationResult:
    X_train, X_test, y_train, y_test = _split(X, y)
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train.astype(int))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
    logger.info("RF fitted: %d train / %d test; class balance=%.2f",
                len(X_train), len(X_test), float(y_train.mean()))
    return ClassificationResult(model, X_train, X_test, y_train, y_test, y_pred, y_proba)


def fit_naive_bayes(X: pd.DataFrame, y: pd.Series) -> ClassificationResult:
    X_train, X_test, y_train, y_test = _split(X, y)
    model = GaussianNB()
    model.fit(X_train, y_train.astype(int))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
    logger.info("NB fitted: %d train / %d test", len(X_train), len(X_test))
    return ClassificationResult(model, X_train, X_test, y_train, y_test, y_pred, y_proba)


def feature_importance(res: ClassificationResult, top_n: int = 15) -> pd.DataFrame:
    """Return sorted feature importance from a fitted RandomForestClassifier."""
    model = res.model
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    importances = pd.DataFrame({
        "feature": res.X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return importances
