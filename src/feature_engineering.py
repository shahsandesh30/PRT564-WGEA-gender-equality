"""
feature_engineering.py — targets and encoded feature matrix.

Owner: Aadarsh Ghimire (Preprocessing & Research Design Lead)

Creates the continuous and binary targets declared in Assessment 1 Sections 4.2
and 4.3, and returns the encoded X matrix ready for scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EMPLOYER_SIZE_ORDER
from .utils import get_logger

logger = get_logger(__name__)


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add prop_women_mgmt, prop_women_overall, and women_mgmt_high columns."""
    df = df.copy()
    total = df["women_total"] + df["men_total"]
    mgmt_total = df["women_mgmt"] + df["men_mgmt"]

    df["prop_women_overall"] = np.where(total > 0, df["women_total"] / total, np.nan)
    df["prop_women_mgmt"] = np.where(mgmt_total > 0, df["women_mgmt"] / mgmt_total, np.nan)

    # Binary target for RQ2 — drop rows where prop_women_mgmt is NaN before splitting
    valid = df["prop_women_mgmt"].dropna()
    if len(valid) > 0:
        median_val = valid.median()
        df["women_mgmt_high"] = (df["prop_women_mgmt"] > median_val).astype("Int64")
        logger.info("RQ2 median split at prop_women_mgmt=%.3f (High vs Low)", median_val)
    else:
        df["women_mgmt_high"] = pd.NA

    return df


def encode_features(
    df: pd.DataFrame,
    target_reg: str = "prop_women_mgmt",
    target_cls: str = "women_mgmt_high",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Return X, y_reg, y_cls, and the list of feature names.

    - Ordinal encodes employer_size (per EMPLOYER_SIZE_ORDER).
    - One-hot encodes anzsic_division.
    - Keeps binary policy flags as-is.
    - Drops employers with missing regression target.
    """
    df = df.dropna(subset=[target_reg]).copy()

    # Ordinal encode employer_size
    size_map = {s: i for i, s in enumerate(EMPLOYER_SIZE_ORDER)}
    df["employer_size_ord"] = df["employer_size"].map(size_map)
    if df["employer_size_ord"].isna().any():
        unknown = df.loc[df["employer_size_ord"].isna(), "employer_size"].unique()
        logger.warning("Unknown employer_size values set to -1: %s", unknown)
        df["employer_size_ord"] = df["employer_size_ord"].fillna(-1).astype(int)

    # One-hot encode anzsic_division
    division_dummies = pd.get_dummies(
        df["anzsic_division"], prefix="div", drop_first=True, dtype=int
    )

    # Policy / context features
    policy_cols = [c for c in df.columns if c.startswith(("has_", "offers_", "took_"))]
    external_numeric = [c for c in df.columns if c in ("industry_pay_gap",)]

    feature_cols = ["employer_size_ord"] + policy_cols + external_numeric
    X = pd.concat([df[feature_cols].reset_index(drop=True),
                   division_dummies.reset_index(drop=True)], axis=1)
    # Ensure numeric dtype for model
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    y_reg = df[target_reg].reset_index(drop=True)
    y_cls = df[target_cls].astype("Int64").reset_index(drop=True)

    logger.info("Encoded X: shape=%s; y_reg=%d; y_cls pos rate=%.2f",
                X.shape, len(y_reg), float(y_cls.mean()) if len(y_cls) else 0.0)
    return X, y_reg, y_cls, list(X.columns)
