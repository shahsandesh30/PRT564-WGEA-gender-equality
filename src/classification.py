"""
classification.py — RQ2 Gaussian Naive Bayes + majority-class baseline.

Owner: Sandesh Shahi (Analysis Lead)

Naive Bayes is the workingmodel; 
the majority-class baseline is the comparator for the paired-CV
t-test that satisfies Assessment 2 rubric item 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
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


def fit_naive_bayes(X: pd.DataFrame, y: pd.Series) -> ClassificationResult:
    """Fit Gaussian Naive Bayes with a stratified 80/20 split."""
    X_train, X_test, y_train, y_test = _split(X, y)
    model = GaussianNB()
    model.fit(X_train, y_train.astype(int))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
    logger.info("NB fitted: %d train / %d test; class balance=%.2f",
                len(X_train), len(X_test), float(y_train.mean()))
    return ClassificationResult(model, X_train, X_test, y_train, y_test, y_pred, y_proba)


def majority_class_baseline() -> DummyClassifier:
    """Return an unfit DummyClassifier that always predicts the training majority class.

    Used as the comparator in the paired-CV t-test for Naive Bayes (RQ2).
    """
    return DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
