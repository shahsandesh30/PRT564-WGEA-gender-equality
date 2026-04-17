"""
utils.py — small shared helpers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd


def get_logger(name: str = "wgea") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_checkpoint(obj, path: Path) -> Path:
    """Pickle any Python object to a checkpoint file (used between notebooks)."""
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_checkpoint(path: Path):
    """Load a pickled checkpoint file."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
