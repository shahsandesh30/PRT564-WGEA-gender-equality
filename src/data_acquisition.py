"""
data_acquisition.py — load.

Owner: Shuvechchha Pun (Data Acquisition & Context Lead)

Corresponds to Assessment 1 Section 2 (Dataset Description) and Assessment 2
rubric requirement 1 (data collection — potentially from multiple sources).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    ACTIVE_DATA_DIR,
    WGEA_FILES,
)
from .utils import get_logger

logger = get_logger(__name__)


def load_wgea(data_dir: Path = ACTIVE_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all 8 WGEA CSVs into a dict keyed by short name."""
    logger.info("Loading WGEA CSVs from %s", data_dir)
    data: dict[str, pd.DataFrame] = {}
    for short_name, filename in WGEA_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing WGEA file: {path}")
        df = pd.read_csv(path, low_memory=False)
        logger.info("  %-45s %6d rows  %3d cols", short_name, len(df), df.shape[1])
        data[short_name] = df
    return data


def validate(data: dict[str, pd.DataFrame]) -> None:
    """Sanity checks on the loaded WGEA data."""
    required_keys = {"employer_abn", "reporting_year", "anzsic_division", "employer_size"}
    for name, df in data.items():
        missing = required_keys - set(df.columns)
        if missing and name != "questionnaire_catalogue":
            # catalogue has a different schema
            raise ValueError(f"{name} missing columns: {missing}")

    # Drop rows with a null employer_abn across all tables that carry the column,
    # and warn so the caller knows the data was trimmed.
    for name, df in data.items():
        if "employer_abn" in df.columns:
            null_count = df["employer_abn"].isna().sum()
            if null_count:
                logger.warning(
                    "%s: dropping %d rows with null employer_abn (%.2f%%)",
                    name, null_count, 100 * null_count / len(df),
                )
                data[name] = df.dropna(subset=["employer_abn"]).reset_index(drop=True)

    wc = data["workforce_composition"]
    logger.info("Validation passed: %d unique employers in workforce_composition",
                wc["employer_abn"].nunique())
