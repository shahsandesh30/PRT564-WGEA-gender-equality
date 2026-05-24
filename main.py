"""
main.py — end-to-end WGEA gender-equality analysis pipeline.

Usage:
    python main.py                 # runs on the full Dataset/wgea_public_dataset_2025
    USE_SAMPLE=1 python main.py    # runs on Dataset/wgea_sample_5rows for a fast wiring check

Produces figures in outputs/figures/ and metric tables in outputs/tables/,
which the Group 9 presentation draws from.

Pipeline stages (owner in brackets):
  1. Data acquisition            [Shuvechchha]
  2. Preprocessing + integration [Aadarsh]
  3. Feature engineering         [Aadarsh]
  4. EDA + assumption checks     [Pujan]
  5. Regression  (RQ1)           [Sandesh]
  6. Classification (RQ2)        [Sandesh]
  7. Paired-CV statistical tests [Sandesh]
  8. Visualisation + reporting   [Pujan]
"""

from __future__ import annotations

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.naive_bayes import GaussianNB

from src import (
    classification,
    data_acquisition,
    eda,
    evaluation,
    feature_engineering,
    preprocessing,
    regression,
    reporting,
    visualisation,
)
from src.config import RANDOM_SEED
from src.utils import get_logger

logger = get_logger("main")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Data acquisition
    # ------------------------------------------------------------------
    data = data_acquisition.load_wgea()
    abs_df = data_acquisition.load_external_abs()
    data_acquisition.validate(data)

    # ------------------------------------------------------------------
    # 2. Preprocessing + heterogeneous integration
    # ------------------------------------------------------------------
    master = preprocessing.build_employer_master(data)
    master = preprocessing.merge_questionnaires(master, data)
    master = preprocessing.integrate_external(master, abs_df)
    master = preprocessing.handle_missing(master)

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    master = feature_engineering.compute_targets(master)
    preprocessing.save_processed(master)

    X, y_reg, y_cls, feature_names = feature_engineering.encode_features(master)

    # ------------------------------------------------------------------
    # 4. EDA + assumption diagnostics
    # ------------------------------------------------------------------
    eda.plot_gender_composition_by_division(master)
    eda.plot_gender_composition_by_size(master)
    eda.plot_policy_vs_workforce_women(master)
    eda.plot_correlation_matrix(X, y_reg)
    vif_df = eda.compute_vif(X)
    reporting.write_vif(vif_df)

    # ------------------------------------------------------------------
    # 5. Regression (RQ1)
    # ------------------------------------------------------------------
    reg_res = regression.fit_linear_regression(X, y_reg)
    regression.diagnostics(reg_res)
    reg_metrics = evaluation.regression_metrics(
        reg_res.y_test.values, reg_res.y_pred_test, n_features=X.shape[1]
    )
    logger.info("RQ1 metrics: %s", reg_metrics)

    visualisation.plot_regression_coefficients(reg_res.model, feature_names)
    visualisation.plot_predicted_vs_actual(reg_res.y_test.values, reg_res.y_pred_test)

    # ------------------------------------------------------------------
    # 6. Classification (RQ2)
    # ------------------------------------------------------------------
    # Drop any rows where y_cls is missing (pd.NA) before classification
    mask_cls = y_cls.notna()
    X_cls = X.loc[mask_cls].reset_index(drop=True)
    y_cls_clean = y_cls.loc[mask_cls].astype(int).reset_index(drop=True)

    rf_res = classification.fit_random_forest(X_cls, y_cls_clean)
    nb_res = classification.fit_naive_bayes(X_cls, y_cls_clean)

    rf_metrics = evaluation.classification_metrics(
        rf_res.y_test.values, rf_res.y_pred_test, rf_res.y_proba_test
    )
    nb_metrics = evaluation.classification_metrics(
        nb_res.y_test.values, nb_res.y_pred_test, nb_res.y_proba_test
    )
    logger.info("RQ2 RF metrics: %s", rf_metrics)
    logger.info("RQ2 NB metrics: %s", nb_metrics)

    importances = classification.feature_importance(rf_res)
    reporting.write_feature_importance(importances)
    visualisation.plot_feature_importance(importances)
    visualisation.plot_confusion_matrix(evaluation.confusion(rf_res.y_test.values, rf_res.y_pred_test))

    # ------------------------------------------------------------------
    # 7. Paired-CV t-tests — statistical significance
    # ------------------------------------------------------------------
    # RQ1: OLS vs Ridge (regression, higher R² better)
    ttest_reg = evaluation.paired_t_test_cv(
        LinearRegression(),
        Ridge(alpha=1.0, random_state=RANDOM_SEED),
        X, y_reg,
        scoring_fn=r2_score,
        stratify=False,
        name_a="OLS",
        name_b="Ridge",
    )

    # RQ2: Random Forest vs Naive Bayes (classification, higher accuracy better)
    ttest_cls = evaluation.paired_t_test_cv(
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=RANDOM_SEED, n_jobs=-1),
        GaussianNB(),
        X_cls, y_cls_clean,
        scoring_fn=accuracy_score,
        stratify=True,
        name_a="RandomForest",
        name_b="NaiveBayes",
    )

    # ------------------------------------------------------------------
    # 8. Reporting
    # ------------------------------------------------------------------
    reporting.write_metrics_summary(reg_metrics, rf_metrics, nb_metrics)
    reporting.write_ttest_results(ttest_reg, ttest_cls)

    logger.info("Pipeline complete. See outputs/figures and outputs/tables for artefacts.")


if __name__ == "__main__":
    main()
