"""
eda.py — exploratory data analysis and assumption diagnostics.

Owner: Pujan Dey (Visualisation & Reporting Lead)

Covers EDA and supports the linear
regression assumption checks.
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


# ---------------------------------------------------------------------------
# RQ3 — gender composition by division / size
# ---------------------------------------------------------------------------
def plot_gender_composition_by_division(df: pd.DataFrame, out: Path = FIG_DIR) -> Path:
    """Horizontal bar chart: mean proportion of women overall, by ANZSIC division."""
    agg = (
        df.groupby("anzsic_division")["prop_women_overall"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    agg.plot(kind="barh", ax=ax, color="steelblue")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="Parity (0.5)")
    ax.set_xlabel("Mean proportion of women (overall workforce)")
    ax.set_ylabel("ANZSIC Division")
    ax.set_title("RQ3 — Gender composition by industry division")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out / "rq3_women_share_by_division.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path


def plot_gender_composition_by_size(df: pd.DataFrame, out: Path = FIG_DIR) -> Path:
    """Bar chart: mean proportion of women overall, by employer size band."""
    agg = df.groupby("employer_size")["prop_women_overall"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    agg.plot(kind="bar", ax=ax, color="teal")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Parity")
    ax.set_ylabel("Mean proportion of women")
    ax.set_xlabel("Employer size band")
    ax.set_title("RQ3 — Gender composition by employer size")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    path = out / "rq3_women_share_by_size.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path


# ---------------------------------------------------------------------------
# RQ4 — policies vs women-in-workforce
# ---------------------------------------------------------------------------
def plot_policy_vs_workforce_women(df: pd.DataFrame, out: Path = FIG_DIR) -> Path:
    """Box plots of prop_women_overall split by key policy flags."""
    policy_flags = [c for c in ("has_employer_paid_parental_leave", "offers_flexible_work",
                                "has_formal_dni_policy", "took_action_on_pay_gap",
                                "has_domestic_violence_policy") if c in df.columns]
    if not policy_flags:
        logger.warning("No policy flags found for RQ4 plot.")
        return out / "rq4_policy_vs_women_SKIPPED.png"

    fig, axes = plt.subplots(1, len(policy_flags), figsize=(4 * len(policy_flags), 4.5), sharey=True)
    if len(policy_flags) == 1:
        axes = [axes]
    for ax, flag in zip(axes, policy_flags):
        sns.boxplot(data=df, x=flag, y="prop_women_overall", ax=ax, palette="Set2")
        ax.set_title(flag.replace("_", " "))
        ax.set_xlabel("0 = No   1 = Yes")
        ax.set_ylabel("Prop. women overall")
    fig.suptitle("RQ4 — Women-in-workforce by policy adoption", y=1.02)
    fig.tight_layout()
    path = out / "rq4_policy_vs_women.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path


# ---------------------------------------------------------------------------
# Correlation matrix + VIF (assumption diagnostics for RQ1)
# ---------------------------------------------------------------------------
def plot_correlation_matrix(X: pd.DataFrame, y: pd.Series, out: Path = FIG_DIR) -> Path:
    df = X.copy()
    df["prop_women_mgmt"] = y.values
    # Pick numeric columns only and drop constants
    num = df.select_dtypes(include=[np.number])
    num = num.loc[:, num.nunique() > 1]
    # Limit to top features by |corr with target| to keep plot readable
    corr_with_target = num.corrwith(num["prop_women_mgmt"]).abs().sort_values(ascending=False)
    keep = corr_with_target.head(15).index
    corr = num[keep].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap (top 15 features + target)")
    fig.tight_layout()
    path = out / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor for each feature — multicollinearity check."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        logger.warning("statsmodels not installed — skipping VIF.")
        return pd.DataFrame(columns=["feature", "vif"])

    Xn = X.select_dtypes(include=[np.number]).copy()
    Xn = Xn.loc[:, Xn.nunique() > 1]
    Xn = Xn.assign(const=1.0)
    rows = []
    for i, col in enumerate(Xn.columns):
        if col == "const":
            continue
        try:
            vif = variance_inflation_factor(Xn.values, i)
        except Exception:
            vif = np.nan
        rows.append({"feature": col, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
