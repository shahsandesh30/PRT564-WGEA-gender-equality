"""
a4.py — Assessment 4 classification pipeline.

Owner: Group 9 (Analysis Lead: Sandesh Shahi).

Self-contained module that produces every figure, table and metric the A4
report needs. Designed so that running `build_a4_artifacts(...)` once will
populate outputs/a4/ with everything required for the writeup.

Classification problem (A4)
---------------------------
Binary: `under_represented_in_mgmt` — does this employer have a women-in-
management proportion strictly below the median of its ANZSIC division
(i.e. a within-industry under-performer)?

Why this target rather than the A2 global median split?
- Industry-relative; controls for the fact that some industries (e.g. mining,
  construction) have structurally low women representation across all
  employers.
- Directly actionable for the regulator (WGEA): flags employers lagging
  behind their peers, not their cross-economy comparators.

Models compared
---------------
Random Forest, Gaussian Naive Bayes (carried from A2), Logistic Regression
and Decision Tree (new in A4). All four are tuned with GridSearchCV using
5-fold StratifiedKFold and the same fixed random seed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats

from .config import CV_FOLDS, OUTPUTS_DIR, RANDOM_SEED, TEST_SIZE
from .utils import get_logger, save_table

logger = get_logger(__name__)

A4_DIR = OUTPUTS_DIR / "a4"
A4_FIG = A4_DIR / "figures"
A4_TAB = A4_DIR / "tables"
A4_MODEL = A4_DIR / "models"
for _d in (A4_FIG, A4_TAB, A4_MODEL):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Target construction
# ---------------------------------------------------------------------------
def construct_industry_relative_target(master: pd.DataFrame) -> pd.DataFrame:
    """Add `prop_women_mgmt`, `industry_median_pwm`, `under_represented_in_mgmt`.

    Drops employers without a computable proportion (zero managers).
    """
    df = master.copy()
    mgmt_total = df["women_mgmt"] + df["men_mgmt"]
    df["prop_women_mgmt"] = np.where(mgmt_total > 0, df["women_mgmt"] / mgmt_total, np.nan)

    before = len(df)
    df = df.dropna(subset=["prop_women_mgmt", "anzsic_division"]).reset_index(drop=True)
    logger.info("Target construction: dropped %d employers with no managers; %d remain",
                before - len(df), len(df))

    industry_med = (
        df.groupby("anzsic_division")["prop_women_mgmt"].median().rename("industry_median_pwm")
    )
    df = df.merge(industry_med, on="anzsic_division", how="left")
    df["under_represented_in_mgmt"] = (df["prop_women_mgmt"] < df["industry_median_pwm"]).astype(int)

    class_balance = float(df["under_represented_in_mgmt"].mean())
    logger.info("Class balance (positive=under-represented): %.3f", class_balance)
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
SIZE_ORDER = ["<250", "250- 499", "500- 999", "1000- 4999", "5000+"]


def add_a4_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `policy_count`, `log_total_employees`, `size_x_division_women_share`."""
    df = df.copy()
    policy_cols = [c for c in df.columns if c.startswith(("has_", "offers_", "took_"))]
    df["policy_count"] = df[policy_cols].fillna(0).sum(axis=1).astype(int)

    total = (df["women_total"] + df["men_total"]).clip(lower=0)
    df["log_total_employees"] = np.log1p(total)

    size_map = {s: i for i, s in enumerate(SIZE_ORDER)}
    df["employer_size_ord"] = df["employer_size"].map(size_map).fillna(-1).astype(int)

    overall_total = (df["women_total"] + df["men_total"]).replace(0, np.nan)
    df["prop_women_overall"] = (df["women_total"] / overall_total).fillna(0)
    div_means = df.groupby("anzsic_division")["prop_women_overall"].mean()
    df["division_mean_women_share"] = df["anzsic_division"].map(div_means)
    df["size_x_division_women_share"] = df["employer_size_ord"] * df["division_mean_women_share"]
    return df


def encode_features_a4(df: pd.DataFrame, target: str = "under_represented_in_mgmt"):
    """Return (X, y, feature_names). One-hot for division, ordinal for size."""
    policy_cols = [c for c in df.columns if c.startswith(("has_", "offers_", "took_"))]
    extra_numeric = ["employer_size_ord", "policy_count", "log_total_employees",
                     "division_mean_women_share", "size_x_division_women_share"]
    division_dummies = pd.get_dummies(df["anzsic_division"], prefix="div", drop_first=True, dtype=int)

    feature_cols = policy_cols + extra_numeric
    X = pd.concat(
        [df[feature_cols].reset_index(drop=True), division_dummies.reset_index(drop=True)],
        axis=1,
    )
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[target].astype(int).reset_index(drop=True)
    logger.info("Encoded A4 features: X=%s, positive rate=%.3f", X.shape, float(y.mean()))
    return X, y, list(X.columns)


# ---------------------------------------------------------------------------
# 3. EDA — report figures
# ---------------------------------------------------------------------------
def plot_class_balance(df: pd.DataFrame, fname: str = "a4_class_balance.png") -> Path:
    counts = df["under_represented_in_mgmt"].value_counts().sort_index()
    labels = ["At or above industry median", "Below industry median"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_ylabel("Number of employers")
    ax.set_title("Class balance — under-representation in management")
    fig.tight_layout()
    out = A4_FIG / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_pwm_by_division(df: pd.DataFrame, fname: str = "a4_pwm_by_division.png") -> Path:
    order = df.groupby("anzsic_division")["prop_women_mgmt"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df.loc[df["anzsic_division"] == d, "prop_women_mgmt"].values for d in order]
    ax.boxplot(data, vert=False, labels=order, showfliers=False)
    ax.set_xlabel("Proportion of women in management")
    ax.set_title("Women-in-management by ANZSIC division (industry medians)")
    fig.tight_layout()
    out = A4_FIG / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series,
                             fname: str = "a4_correlation_heatmap.png") -> Path:
    keep = [c for c in X.columns if not c.startswith("div_")][:15]
    df_corr = X[keep].copy()
    df_corr["target"] = y.values
    corr = df_corr.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Feature correlation heatmap (non-division features)")
    fig.tight_layout()
    out = A4_FIG / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_outliers(df: pd.DataFrame, fname: str = "a4_outliers.png") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(df["prop_women_mgmt"].dropna(), vert=True)
    axes[0].set_title("prop_women_mgmt (IQR / outliers)")
    axes[1].hist(np.log1p(df["women_total"] + df["men_total"]), bins=40, color="#55A868")
    axes[1].set_title("log(total employees) distribution")
    fig.tight_layout()
    out = A4_FIG / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. Models + hyperparameter tuning
# ---------------------------------------------------------------------------
def model_factory() -> dict[str, tuple[Pipeline, dict]]:
    """Return {name: (pipeline, param_grid)} for all four models."""
    return {
        "RandomForest": (
            Pipeline([("clf", RandomForestClassifier(
                random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced"))]),
            {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [None, 8, 16],
                "clf__min_samples_leaf": [1, 5],
            },
        ),
        "NaiveBayes": (
            Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", GaussianNB())]),
            {"clf__var_smoothing": np.logspace(-11, -7, 5).tolist()},
        ),
        "LogisticRegression": (
            Pipeline([("scaler", StandardScaler(with_mean=False)),
                      ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))]),
            {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "DecisionTree": (
            Pipeline([("clf", DecisionTreeClassifier(
                random_state=RANDOM_SEED, class_weight="balanced"))]),
            {
                "clf__max_depth": [3, 5, 10, None],
                "clf__min_samples_split": [2, 10],
                "clf__criterion": ["gini", "entropy"],
            },
        ),
    }


@dataclass
class FittedModel:
    name: str
    estimator: Pipeline
    best_params: dict
    cv_best_f1: float
    y_pred: np.ndarray
    y_proba: np.ndarray
    test_metrics: dict
    cv_metrics: dict
    cv_f1_scores: list


def tune_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict[str, FittedModel]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results: dict[str, FittedModel] = {}

    for name, (pipe, grid) in model_factory().items():
        logger.info("Tuning %s ...", name)
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)

        est = gs.best_estimator_
        y_pred = est.predict(X_test)
        y_proba = est.predict_proba(X_test)[:, 1]
        test_metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_proba),
        }
        # Full-data CV (refit on best params) for reported mean ± std
        cv_metrics, cv_f1_scores = _cv_metrics(est, X, y, cv)

        results[name] = FittedModel(
            name=name, estimator=est, best_params=gs.best_params_,
            cv_best_f1=float(gs.best_score_), y_pred=y_pred, y_proba=y_proba,
            test_metrics=test_metrics, cv_metrics=cv_metrics, cv_f1_scores=cv_f1_scores,
        )
        logger.info("  best F1 (CV)=%.4f params=%s", gs.best_score_, gs.best_params_)
    return results


def _cv_metrics(est, X, y, cv) -> tuple[dict, list]:
    """Per-fold mean ± std for Accuracy/Precision/Recall/F1/ROC-AUC."""
    out = {}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        scores = cross_val_score(est, X, y, scoring=metric, cv=cv, n_jobs=-1)
        out[metric] = {"mean": float(scores.mean()), "std": float(scores.std(ddof=1))}
        if metric == "f1":
            f1_scores = scores.tolist()
    return out, f1_scores


# ---------------------------------------------------------------------------
# 5. Evaluation outputs
# ---------------------------------------------------------------------------
def metrics_summary_table(results: dict[str, FittedModel]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        row = {"Model": name}
        row.update({f"Test_{k}": v for k, v in r.test_metrics.items()})
        for m, s in r.cv_metrics.items():
            row[f"CV_{m}_mean"] = s["mean"]
            row[f"CV_{m}_std"] = s["std"]
        row["CV_best_F1_tuning"] = r.cv_best_f1
        row["best_params"] = json.dumps(r.best_params)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def plot_roc_curves(results: dict[str, FittedModel], y_test: pd.Series,
                    fname: str = "a4_roc_curves.png") -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r.y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={r.test_metrics['ROC_AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — under-representation classifier")
    ax.legend()
    fig.tight_layout()
    out = A4_FIG / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion_matrices(results: dict[str, FittedModel], y_test: pd.Series,
                            fname: str = "a4_confusion_matrices.png") -> Path:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r.y_pred)
        ConfusionMatrixDisplay(cm, display_labels=[">= median", "< median"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name)
    fig.suptitle("Confusion matrices (hold-out test set)")
    fig.tight_layout()
    out = A4_FIG / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_metric_bars(results: dict[str, FittedModel],
                     fname: str = "a4_metric_comparison.png") -> Path:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    names = list(results.keys())
    width = 0.15
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        vals = [results[name].test_metrics[m] for m in metrics]
        ax.bar(x + i * width, vals, width=width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (hold-out test)")
    ax.set_title("Model comparison across metrics")
    ax.legend()
    fig.tight_layout()
    out = A4_FIG / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_feature_importance(results: dict[str, FittedModel], feature_names: list[str],
                            fname: str = "a4_feature_importance.png", top_n: int = 15) -> Path:
    rf = results["RandomForest"].estimator.named_steps["clf"]
    imp = pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#4C72B0")
    ax.set_xlabel("Importance (Random Forest)")
    ax.set_title(f"Top {top_n} features driving under-representation prediction")
    fig.tight_layout()
    out = A4_FIG / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def pairwise_ttests(results: dict[str, FittedModel]) -> pd.DataFrame:
    """Paired t-test on per-fold F1 scores for every model pair."""
    names = list(results.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sa = np.array(results[a].cv_f1_scores)
            sb = np.array(results[b].cv_f1_scores)
            t, p = stats.ttest_rel(sa, sb)
            rows.append({
                "model_a": a, "model_b": b,
                "mean_f1_a": float(sa.mean()), "mean_f1_b": float(sb.mean()),
                "t_statistic": float(t), "p_value": float(p),
                "significant_at_0.05": bool(p < 0.05),
                "better": a if sa.mean() >= sb.mean() else b,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. End-to-end driver
# ---------------------------------------------------------------------------
@dataclass
class A4Artifacts:
    target_table: Path
    figures: dict[str, Path] = field(default_factory=dict)
    tables: dict[str, Path] = field(default_factory=dict)
    models: dict[str, Path] = field(default_factory=dict)
    summary_md: Path | None = None


def build_a4_artifacts(master: pd.DataFrame) -> A4Artifacts:
    """One-call driver that produces every report-ready artifact under outputs/a4/."""
    import pickle

    art = A4Artifacts(target_table=A4_TAB / "a4_dataset.parquet")

    # Step 1+2: target + features
    df = construct_industry_relative_target(master)
    df = add_a4_features(df)
    try:
        df.to_parquet(art.target_table, index=False)
    except Exception:
        art.target_table = art.target_table.with_suffix(".csv")
        df.to_csv(art.target_table, index=False)

    X, y, feat_names = encode_features_a4(df)
    save_table(pd.DataFrame({"feature": feat_names}), A4_TAB / "a4_feature_names.csv")
    art.tables["features"] = A4_TAB / "a4_feature_names.csv"

    # Step 3: EDA figures
    art.figures["class_balance"] = plot_class_balance(df)
    art.figures["pwm_by_division"] = plot_pwm_by_division(df)
    art.figures["correlation"] = plot_correlation_heatmap(X, y)
    art.figures["outliers"] = plot_outliers(df)

    # Step 4: tuned models
    results = tune_and_evaluate(X, y)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    summary = metrics_summary_table(results)
    save_table(summary, A4_TAB / "a4_metrics_summary.csv")
    art.tables["metrics_summary"] = A4_TAB / "a4_metrics_summary.csv"

    ttests = pairwise_ttests(results)
    save_table(ttests, A4_TAB / "a4_pairwise_ttests.csv")
    art.tables["pairwise_ttests"] = A4_TAB / "a4_pairwise_ttests.csv"

    best_params_df = pd.DataFrame([
        {"model": n, "best_params": json.dumps(r.best_params), "cv_best_f1": r.cv_best_f1}
        for n, r in results.items()
    ])
    save_table(best_params_df, A4_TAB / "a4_best_hyperparameters.csv")
    art.tables["best_hyperparameters"] = A4_TAB / "a4_best_hyperparameters.csv"

    # Step 5: comparison plots
    art.figures["roc"] = plot_roc_curves(results, y_test)
    art.figures["confusion"] = plot_confusion_matrices(results, y_test)
    art.figures["metric_bars"] = plot_metric_bars(results)
    art.figures["feature_importance"] = plot_feature_importance(results, feat_names)

    # Step 6: persist fitted estimators
    for name, r in results.items():
        path = A4_MODEL / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(r.estimator, f)
        art.models[name] = path

    # Step 7: report-ready summary markdown (numbers auto-injected)
    art.summary_md = _write_summary_markdown(df, X, y, results, ttests)
    return art


def _write_summary_markdown(df, X, y, results, ttests) -> Path:
    best = max(results.values(), key=lambda r: r.test_metrics["F1"])
    lines = [
        "# A4 results — auto-generated summary",
        "",
        f"- Employers retained after target construction: **{len(df):,}**",
        f"- Features used: **{X.shape[1]}** (policy flags + engineered + one-hot division)",
        f"- Class balance (positive = below industry median): "
        f"**{float(y.mean()):.1%}** of {len(y):,} employers",
        "",
        "## Hold-out test-set metrics",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        m = r.test_metrics
        lines.append(
            f"| {name} | {m['Accuracy']:.3f} | {m['Precision']:.3f} | "
            f"{m['Recall']:.3f} | {m['F1']:.3f} | {m['ROC_AUC']:.3f} |"
        )

    lines += [
        "",
        f"**Best model on hold-out F1:** {best.name} "
        f"(F1={best.test_metrics['F1']:.3f}, ROC-AUC={best.test_metrics['ROC_AUC']:.3f}).",
        "",
        "## 5-fold CV (mean ± std)",
        "",
        "| Model | F1 | ROC-AUC | Recall |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        f1 = r.cv_metrics["f1"]; auc = r.cv_metrics["roc_auc"]; rec = r.cv_metrics["recall"]
        lines.append(
            f"| {name} | {f1['mean']:.3f} ± {f1['std']:.3f} | "
            f"{auc['mean']:.3f} ± {auc['std']:.3f} | "
            f"{rec['mean']:.3f} ± {rec['std']:.3f} |"
        )

    lines += ["", "## Pairwise paired t-tests on per-fold F1", ""]
    lines.append("| A | B | mean F1 A | mean F1 B | t | p | sig (0.05) | better |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in ttests.iterrows():
        lines.append(
            f"| {row['model_a']} | {row['model_b']} | {row['mean_f1_a']:.3f} | "
            f"{row['mean_f1_b']:.3f} | {row['t_statistic']:.2f} | "
            f"{row['p_value']:.4f} | {row['significant_at_0.05']} | {row['better']} |"
        )

    path = A4_TAB / "a4_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
