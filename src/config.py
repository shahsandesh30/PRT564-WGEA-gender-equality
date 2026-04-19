"""
config.py — paths, seeds, constants.

Owner: Shuvechchha Pun (Data Acquisition & Context Lead)

Single source of truth for file paths and pipeline constants.
Flip USE_SAMPLE to run on the 5-row sample folder for fast wiring tests.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "Dataset"
DATASET_FULL = DATASET_ROOT / "wgea_public_dataset_2025"
DATASET_SAMPLE = DATASET_ROOT / "wgea_sample_5rows"

DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED = DATA_DIR / "processed"
CHECKPOINT_DIR = DATA_PROCESSED / "checkpoints"  # notebook-to-notebook handoffs

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
TABLE_DIR = OUTPUTS_DIR / "tables"

# ---------------------------------------------------------------------------
# Runtime switches
# ---------------------------------------------------------------------------
# Toggle with env var: USE_SAMPLE=1 python main.py
USE_SAMPLE: bool = os.environ.get("USE_SAMPLE", "0") == "1"

ACTIVE_DATA_DIR: Path = DATASET_SAMPLE if USE_SAMPLE else DATASET_FULL

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Dataset schema — file registry
# ---------------------------------------------------------------------------
# Short-name -> filename (same in both sample and full dirs)
WGEA_FILES: dict[str, str] = {
    "workforce_composition": "wgea_workforce_composition_2025.csv",
    "workforce_management_statistics": "wgea_workforce_management_statistics_2025.csv",
    "questionnaire_workplace_overview": "wgea_questionnaire_workplace_overview_2025.csv",
    "questionnaire_action_on_gender_equality": "wgea_questionnaire_action_on_gender_equality_2025.csv",
    "questionnaire_employee_support": "wgea_questionnaire_employee_support_2025.csv",
    "questionnaire_flexible_work": "wgea_questionnaire_flexible_work_2025.csv",
    "questionnaire_harm_prevention": "wgea_questionnaire_harm_prevention_2025.csv",
    "questionnaire_catalogue": "wgea_questionnaire_catalogue_2025.csv",
}


# ---------------------------------------------------------------------------
# Categorical encodings
# ---------------------------------------------------------------------------
# Size bands as they appear verbatim in the WGEA CSVs (no spaces inside the ranges).
EMPLOYER_SIZE_ORDER: list[str] = ["<250", "250-499", "500-999", "1000-4999", "5000+"]

# Questionnaire question *stems* → short binary feature names.
# WGEA's schema splits each Yes/No question into two `question_index` branches:
# `<stem>.Y` (employers who answered Yes) and `<stem>.N` (answered No).
# `_extract_question_flag` uses the stem to read both branches and emit 1/0.
QUESTIONNAIRE_FEATURE_MAP: dict[str, str] = {
    # Workplace overview — D&I policy
    "DnI.FPS": "has_formal_dni_policy",
    # Action on gender equality — remuneration gap analysis actions
    "EAct.Act": "took_action_on_pay_gap",
    # Employee support — employer-funded paid parental leave
    "PPL.RegCarer": "has_employer_paid_parental_leave",
    # Harm prevention — domestic violence formal policy
    "DV.DV": "has_domestic_violence_policy",
}

# Flexible-work question codes — presence of any response means that option is offered.
# We collapse all flexible-work rows into a single binary: at least one option offered.
FLEX_WORK_SECTION_NAME: str = "Flexible work"
