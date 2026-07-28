"""
Full training pipeline for Medical Insurance Cost prediction.

Execution order
---------------
  1. Load raw data
  2. Clean  (preprocessing.clean_dataframe)
  3. Engineer features  (feature_engineering.engineer_features)
  4. Split  (preprocessing.split_data)
  5. Train + evaluate 7 required regression algorithms
  6. Cross-validate each model
  7. Compute learning curves
  8. Select best model (composite rank)
  9. Save best model + metadata to models/
 10. Persist comparison CSV + Markdown report

Run from the project root:
    python -m src.train

Models trained
--------------
  Required (7):
    Linear Regression, Ridge, Lasso,
    Decision Tree, Random Forest, Gradient Boosting, SVR

  Bonus (3):
    XGBoost, Extra Trees, AdaBoost
    (installed automatically if xgboost is present; silently skipped otherwise)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.evaluate import (
    compute_learning_curve,
    evaluate_model,
    generate_comparison_report,
    run_cross_validation,
    save_comparison_csv,
    select_best_model,
)
from src.feature_engineering import engineer_features, get_all_feature_groups
from src.preprocessing import clean_dataframe, get_feature_names_out, split_data
from src.utils import (
    METRICS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    ensure_output_dirs,
    get_logger,
    load_raw_data,
    save_json,
)

log = get_logger("train")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Models that need feature scaling (linear decision boundary or kernel trick)
SCALED_MODELS = {
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Support Vector Regressor",
}


def _get_model_registry() -> dict[str, Any]:
    """
    Return all regression estimators to compare.

    XGBoost is added only when the package is installed — if not present
    the rest of training still runs without errors.
    """
    registry: dict[str, Any] = {
        # ── Required 7 ──────────────────────────────────────────────────────
        "Linear Regression":        LinearRegression(),
        "Ridge Regression":         Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso Regression":         Lasso(alpha=1.0,  random_state=RANDOM_STATE, max_iter=10_000),
        "Decision Tree Regressor":  DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10),
        "Random Forest Regressor":  RandomForestRegressor(
                                        n_estimators=200, random_state=RANDOM_STATE,
                                        max_depth=12, n_jobs=-1,
                                    ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
                                        n_estimators=200, random_state=RANDOM_STATE,
                                        max_depth=4, learning_rate=0.05, subsample=0.8,
                                    ),
        "Support Vector Regressor": SVR(kernel="rbf", C=1000, gamma="scale", epsilon=0.1),

        # ── Bonus ────────────────────────────────────────────────────────────
        "Extra Trees Regressor":    ExtraTreesRegressor(
                                        n_estimators=200, random_state=RANDOM_STATE,
                                        max_depth=12, n_jobs=-1,
                                    ),
        "AdaBoost Regressor":       AdaBoostRegressor(
                                        n_estimators=100, random_state=RANDOM_STATE,
                                        learning_rate=0.1,
                                    ),
    }

    # XGBoost — optional bonus model
    try:
        from xgboost import XGBRegressor
        registry["XGBoost Regressor"] = XGBRegressor(
            n_estimators=200, random_state=RANDOM_STATE,
            max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, n_jobs=-1,
        )
        log.info("XGBoost found — added to registry.")
    except ImportError:
        log.info("XGBoost not installed — skipping bonus model.")

    return registry


# ---------------------------------------------------------------------------
# Pipeline builder  (wraps preprocessor + estimator)
# ---------------------------------------------------------------------------

def _build_pipeline(
    estimator: Any,
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str,
) -> Pipeline:
    """
    Wrap an estimator in a full sklearn Pipeline with preprocessing.

    Scaling is applied only for models that need it (SCALED_MODELS).
    Tree-based models skip StandardScaler — they are scale-invariant and
    scaling adds noise to their split decisions.
    """
    scale = model_name in SCALED_MODELS

    # Numeric branch
    num_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    # Categorical branch
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor",    estimator),
    ])


def _feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Extract transformed feature names from a fitted Pipeline."""
    ct: ColumnTransformer = pipeline.named_steps["preprocessor"]
    names: list[str] = []
    for tname, transformer, cols in ct.transformers_:
        if tname == "num":
            names.extend(cols)
        elif tname == "cat":
            enc: OneHotEncoder = transformer.named_steps["encoder"]
            names.extend(enc.get_feature_names_out(cols).tolist())
    return names


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
    pd.DataFrame,  # full enriched df (for EDA / app)
    list[str],     # numeric feature names
    list[str],     # categorical feature names
]:
    """
    Load → clean → engineer → split.

    Returns train/test splits, full enriched dataframe, and feature name lists.
    """
    log.info("── Step 1/5  Loading raw data ──────────────────────────────")
    raw = load_raw_data()
    log.info("Raw data loaded: %s rows × %s cols", *raw.shape)

    log.info("── Step 2/5  Cleaning ──────────────────────────────────────")
    cleaned = clean_dataframe(raw)

    log.info("── Step 3/5  Feature engineering ───────────────────────────")
    enriched = engineer_features(cleaned)

    # Save enriched dataset for the Streamlit app
    ensure_output_dirs()
    enriched.to_csv(REPORTS_DIR / "cleaned_data.csv", index=False)
    log.info("Enriched dataset saved → outputs/reports/cleaned_data.csv")

    log.info("── Step 4/5  Splitting 80 / 20 ─────────────────────────────")
    X_train, X_test, y_train, y_test = split_data(enriched, test_size=0.20)

    # Derive feature groups from the training split
    num_feats, cat_feats = get_all_feature_groups(X_train)
    log.info("Features — numeric: %d  categorical: %d", len(num_feats), len(cat_feats))

    return X_train, X_test, y_train, y_test, enriched, num_feats, cat_feats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    num_feats: list[str],
    cat_feats: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """
    Train every model in the registry, evaluate, cross-validate.

    Returns
    -------
    metrics_df   : pd.DataFrame — one row per model, all metrics
    fitted_models: dict[name → fitted Pipeline]
    predictions  : dict[name → y_pred array on dollar scale]
    """
    registry = _get_model_registry()
    all_metrics:   list[dict]          = []
    fitted_models: dict[str, Pipeline] = {}
    predictions:   dict[str, np.ndarray] = {}
    cv_results:    dict[str, dict]     = {}
    lc_data:       dict[str, dict]     = {}

    log.info("── Step 5/5  Training %d models ─────────────────────────────", len(registry))
    total = len(registry)

    for idx, (name, estimator) in enumerate(registry.items(), start=1):
        log.info("[%d/%d] %s", idx, total, name)

        pipeline = _build_pipeline(
            clone(estimator), num_feats, cat_feats, name
        )

        metrics, fitted, y_pred = evaluate_model(
            pipeline, X_train, X_test, y_train, y_test,
            model_name=name,
            log_transformed=False,
        )
        all_metrics.append(metrics)
        fitted_models[name] = fitted
        predictions[name]   = y_pred

        # 5-fold cross-validation (fresh clone, no data leakage)
        cv_pipeline = _build_pipeline(clone(estimator), num_feats, cat_feats, name)
        cv_results[name] = run_cross_validation(cv_pipeline, X_train, y_train)
        log.info(
            "  CV R² = %.4f ± %.4f",
            cv_results[name]["r2_mean"],
            cv_results[name]["r2_std"],
        )

        # Learning curve (only on key models to save time)
        if name in {
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "Ridge Regression",
            "Support Vector Regressor",
        }:
            lc_pipeline = _build_pipeline(clone(estimator), num_feats, cat_feats, name)
            lc_data[name] = compute_learning_curve(lc_pipeline, X_train, y_train)

    metrics_df = pd.DataFrame(all_metrics)

    # Persist metrics
    save_comparison_csv(metrics_df)
    save_json(cv_results, METRICS_DIR / "cross_validation.json")
    if lc_data:
        save_json(lc_data, METRICS_DIR / "learning_curves.json")

    return metrics_df, fitted_models, predictions


# ---------------------------------------------------------------------------
# Save best model
# ---------------------------------------------------------------------------

def save_best_model(
    fitted_models: dict[str, Any],
    best_name: str,
    num_feats: list[str],
    cat_feats: list[str],
) -> Path:
    """
    Serialise the winning pipeline to models/best_model.pkl.

    Also writes:
      models/model_metadata.json  — model name, feature lists, target column
      models/feature_info.json    — feature names needed by the Streamlit app
    """
    ensure_output_dirs()

    bundle = {
        "model":    fitted_models[best_name],
        "metadata": {
            "model_name":           best_name,
            "target_column":        TARGET_COLUMN,
            "numeric_features":     num_feats,
            "categorical_features": cat_feats,
        },
    }

    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(bundle, model_path)
    log.info("Best model saved → %s", model_path)

    # Feature info for the prediction/app layer
    feature_info = {
        "best_model":           best_name,
        "numeric_features":     num_feats,
        "categorical_features": cat_feats,
        "all_features":         num_feats + cat_feats,
        "target_column":        TARGET_COLUMN,
    }
    save_json(feature_info,          MODELS_DIR / "feature_info.json")
    save_json(bundle["metadata"],    MODELS_DIR / "model_metadata.json")

    return model_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_training_pipeline() -> dict[str, Any]:
    """
    Execute the complete training pipeline end-to-end and return a summary dict.

    This is the single entry point called by  `python -m src.train`.
    """
    t_start = time.perf_counter()
    ensure_output_dirs()

    # ── Data ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, enriched, num_feats, cat_feats = prepare_data()

    # ── Train ─────────────────────────────────────────────────────────────────
    metrics_df, fitted_models, predictions = train_all_models(
        X_train, X_test, y_train, y_test, num_feats, cat_feats
    )

    # ── Select + report ───────────────────────────────────────────────────────
    best_name = select_best_model(metrics_df)
    generate_comparison_report(metrics_df, best_name)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = save_best_model(fitted_models, best_name, num_feats, cat_feats)

    # ── Print leaderboard ─────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    log.info("\n%s", "=" * 65)
    log.info("  TRAINING COMPLETE  (%.1f s total)", elapsed)
    log.info("=" * 65)

    display = (
        metrics_df
        .sort_values("rmse")
        .assign(
            RMSE  = lambda d: d["rmse"].apply(lambda v: f"${v:,.0f}"),
            MAE   = lambda d: d["mae"].apply(lambda v: f"${v:,.0f}"),
            R2    = lambda d: d["r2"].apply(lambda v: f"{v:.4f}"),
            Time  = lambda d: d["train_time_sec"].apply(lambda v: f"{v:.2f}s"),
        )
        [["model", "RMSE", "MAE", "R2", "Time"]]
        .rename(columns={"model": "Model"})
        .reset_index(drop=True)
    )
    display.index += 1
    log.info("\n%s", display.to_string())
    log.info("\n  ⭐  Best model: %s", best_name)
    log.info("=" * 65 + "\n")

    summary = {
        "best_model":   best_name,
        "model_path":   str(model_path),
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "elapsed_sec":  round(elapsed, 2),
        "metrics":      metrics_df.to_dict(orient="records"),
    }
    save_json(summary, REPORTS_DIR / "training_summary.json")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_training_pipeline()
