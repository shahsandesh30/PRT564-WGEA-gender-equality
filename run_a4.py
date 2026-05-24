"""Driver: build all A4 artifacts in one go.

Usage:
    python run_a4.py            # full dataset
    USE_SAMPLE=1 python run_a4.py  # sample mode (won't yield publishable metrics)
"""
from __future__ import annotations

from src.data_acquisition import load_external_abs, load_wgea, validate
from src.preprocessing import (
    build_employer_master,
    handle_missing,
    integrate_external,
    merge_questionnaires,
)
from src.a4 import build_a4_artifacts


def main() -> None:
    data = load_wgea()
    # Drop rows missing the primary employer key before validation (full dataset has some)
    for key, df in data.items():
        if key != "questionnaire_catalogue" and "employer_abn" in df.columns:
            data[key] = df.dropna(subset=["employer_abn"]).reset_index(drop=True)
    validate(data)
    master = build_employer_master(data)
    master = merge_questionnaires(master, data)
    master = integrate_external(master, load_external_abs())
    master = handle_missing(master)

    art = build_a4_artifacts(master)

    print("\n=== A4 artifacts ===")
    print(f"Target table  : {art.target_table}")
    print("Figures:")
    for k, p in art.figures.items():
        print(f"  {k:25s} {p}")
    print("Tables:")
    for k, p in art.tables.items():
        print(f"  {k:25s} {p}")
    print("Models:")
    for k, p in art.models.items():
        print(f"  {k:25s} {p}")
    print(f"\nReport-ready summary: {art.summary_md}")


if __name__ == "__main__":
    main()
