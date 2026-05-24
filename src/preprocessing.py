"""
preprocessing.py — cleaning, merging, and heterogeneous integration.

Owner: Aadarsh Ghimire (Preprocessing & Research Design Lead)

Corresponds to Assessment 1 Section 4.1 and Assessment 2 rubric requirement 1.
Builds a single employer-level master table keyed on employer_abn, integrating
workforce composition, management movements, and five questionnaire tables.
"""

from __future__ import annotations

import pandas as pd

from .config import (
    DATA_PROCESSED,
    FLEX_WORK_SECTION_NAME,
    QUESTIONNAIRE_FEATURE_MAP,
)
from .utils import get_logger

logger = get_logger(__name__)

# Columns that identify a single employer (one row per employer in the master)
EMPLOYER_ID_COLS = [
    "employer_abn",
    "employer_name",
    "employer_size",
    "anzsic_division",
]


# ---------------------------------------------------------------------------
# Workforce composition → headcount features
# ---------------------------------------------------------------------------
def clean_workforce_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Keep relevant rows, normalise genders, ensure integer headcounts."""
    df = df.copy()
    df = df[df["is_relevant_employer"].astype(str).str.upper() == "TRUE"]
    df["gender"] = df["gender"].astype(str).str.strip()
    df = df[df["gender"].isin(["Women", "Men"])]
    df["n_employees"] = pd.to_numeric(df["n_employees"], errors="coerce").fillna(0).astype(int)
    return df


def _pivot_headcount(df: pd.DataFrame, manager_only: bool) -> pd.DataFrame:
    """Aggregate n_employees by employer × gender, optionally filtering managers."""
    subset = df
    if manager_only:
        # 'Manager' category covers CEO/execs/managers. Non-manager covers everyone else.
        subset = subset[subset["manager_category"].astype(str).str.strip() == "Manager"]
    grouped = (
        subset.groupby(["employer_abn", "gender"])["n_employees"].sum().unstack(fill_value=0)
    )
    # Guarantee both columns exist
    for col in ("Women", "Men"):
        if col not in grouped.columns:
            grouped[col] = 0
    return grouped[["Women", "Men"]]


def build_employer_master(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one-row-per-employer master table with headcount + context cols."""
    wc = clean_workforce_composition(data["workforce_composition"])

    # Employer-level context (take first occurrence — same across all its rows)
    context = (
        wc[EMPLOYER_ID_COLS]
        .drop_duplicates(subset=["employer_abn"])
        .set_index("employer_abn")
    )

    overall = _pivot_headcount(wc, manager_only=False).rename(
        columns={"Women": "women_total", "Men": "men_total"}
    )
    mgmt = _pivot_headcount(wc, manager_only=True).rename(
        columns={"Women": "women_mgmt", "Men": "men_mgmt"}
    )

    master = context.join([overall, mgmt], how="left").fillna(0)

    # Add movements (promotions / resignations) by gender, if file usable
    wms = data.get("workforce_management_statistics")
    if wms is not None and len(wms) > 0:
        wms = wms[wms["is_relevant_employer"].astype(str).str.upper() == "TRUE"].copy()
        wms["gender"] = wms["gender"].astype(str).str.strip()
        wms["n_employees"] = pd.to_numeric(wms["n_employees"], errors="coerce").fillna(0).astype(int)
        wms = wms[wms["gender"].isin(["Women", "Men"])]
        # Normalise movement types (dataset has a typo 'Promtions' in the sample)
        wms["movement_type"] = (
            wms["movement_type"].astype(str).str.strip().str.lower().str.replace("promtions", "promotions")
        )
        wms["movement_key"] = wms["movement_type"].str.extract(r"(promotion|resignation)", expand=False)
        wms = wms.dropna(subset=["movement_key"])
        pivot = (
            wms.groupby(["employer_abn", "movement_key", "gender"])["n_employees"]
            .sum()
            .unstack(["movement_key", "gender"], fill_value=0)
        )
        pivot.columns = [f"{mv}_{g.lower()}" for mv, g in pivot.columns]
        master = master.join(pivot, how="left").fillna(0)

    master = master.reset_index()
    logger.info("Employer master table: %d employers, %d cols", len(master), master.shape[1])
    return master


# ---------------------------------------------------------------------------
# Questionnaires → binary policy features
# ---------------------------------------------------------------------------
def _binary_from_yes(series: pd.Series) -> pd.Series:
    """Map common questionnaire Yes/No responses to 1/0; other → NaN."""
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out[s == "yes"] = 1
    out[s == "no"] = 0
    return out


def _extract_question_flag(df: pd.DataFrame, question_code: str, feature_name: str) -> pd.Series:
    """Pull a single Yes/No question_index into an employer-level binary flag."""
    subset = df[df["question_index"].astype(str).str.strip() == question_code]
    if subset.empty:
        logger.warning("No rows for question_index=%s (feature=%s)", question_code, feature_name)
        return pd.Series(dtype="Int64", name=feature_name)
    binary = _binary_from_yes(subset["response"])
    out = (
        pd.DataFrame({"employer_abn": subset["employer_abn"].values, feature_name: binary.values})
        .groupby("employer_abn")[feature_name]
        .max()  # If multiple rows, 'Yes' dominates
    )
    return out


def _flexible_work_flag(df: pd.DataFrame) -> pd.Series:
    """Any flexible-work option offered → 1 (employer appears in flexible_work table)."""
    if df.empty:
        return pd.Series(dtype="Int64", name="offers_flexible_work")
    offers = df.groupby("employer_abn").size() > 0
    return offers.astype("Int64").rename("offers_flexible_work")


def merge_questionnaires(master: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach binary policy features from the five questionnaire tables."""
    features: list[pd.Series] = []

    # Map each question_index → feature name from the five non-catalogue questionnaires
    questionnaire_tables = [
        "questionnaire_workplace_overview",
        "questionnaire_action_on_gender_equality",
        "questionnaire_employee_support",
        "questionnaire_harm_prevention",
    ]
    for tbl_name in questionnaire_tables:
        df = data[tbl_name]
        for q_code, feat_name in QUESTIONNAIRE_FEATURE_MAP.items():
            if q_code in df["question_index"].astype(str).values:
                features.append(_extract_question_flag(df, q_code, feat_name))

    # Flexible work → single aggregated flag
    features.append(_flexible_work_flag(data["questionnaire_flexible_work"]))

    feat_df = pd.concat(features, axis=1) if features else pd.DataFrame()
    feat_df = feat_df.reset_index().rename(columns={"index": "employer_abn"})
    merged = master.merge(feat_df, on="employer_abn", how="left")
    logger.info("After questionnaire merge: %d cols (%d policy features added)",
                merged.shape[1], feat_df.shape[1] - 1)
    return merged


# ---------------------------------------------------------------------------
# Heterogeneous integration
# ---------------------------------------------------------------------------
def integrate_external(master: pd.DataFrame, abs_df: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join industry-level ABS pay-gap data on anzsic_division."""
    if abs_df is None:
        logger.info("No external ABS data — skipping heterogeneous integration.")
        return master
    if "anzsic_division" not in abs_df.columns:
        logger.warning("External ABS file missing anzsic_division — skipping.")
        return master
    merged = master.merge(abs_df, on="anzsic_division", how="left")
    logger.info("Integrated external ABS features: new cols=%s",
                [c for c in abs_df.columns if c != "anzsic_division"])
    return merged


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing target prerequisites; fill policy flags with 0."""
    df = df.copy()

    # Drop employers with zero total headcount (can't compute proportions)
    total_headcount = df["women_total"] + df["men_total"]
    before = len(df)
    df = df[total_headcount > 0].reset_index(drop=True)
    logger.info("Dropped %d rows with zero headcount; %d remain", before - len(df), len(df))

    # Policy flag columns — fill NaN with 0 (absence of affirmative response)
    policy_cols = [c for c in df.columns if c.startswith(("has_", "offers_", "took_"))]
    for c in policy_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


def save_processed(df: pd.DataFrame) -> None:
    out = DATA_PROCESSED / "employer_level.parquet"
    try:
        df.to_parquet(out, index=False)
        logger.info("Saved processed employer-level table → %s", out)
    except Exception as exc:  # pyarrow not installed etc.
        csv_fallback = out.with_suffix(".csv")
        df.to_csv(csv_fallback, index=False)
        logger.warning("Parquet failed (%s); wrote CSV fallback → %s", exc, csv_fallback)
