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
    Linear Regression, Ridge (RidgeCV), Lasso (LassoCV),
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
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
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
from src.preprocessing import build_preprocessor, clean_dataframe, get_feature_names_out, split_data
from src.utils import (
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    ensure_output_dirs,
    get_logger,
    load_raw_data,
    save_json,
)
from src.visualization import generate_all_eda_figures, generate_all_evaluation_figures

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
    """
    registry: dict[str, Any] = {
        # ── Required 7 ──────────────────────────────────────────────────────
        "Linear Regression": LinearRegression(),

        # RidgeCV searches over 20 log-spaced alphas (0.01 → 1000) using
        # leave-one-out CV internally — zero extra compute vs a fixed alpha.
        "Ridge Regression": RidgeCV(
            alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0,
                    10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
            scoring="neg_root_mean_squared_error",
        ),

        # LassoCV uses coordinate descent with 5-fold CV to find the best
        # alpha from 100 automatically generated candidates on the regularisation
        # path — guarantees the shrinkage level is data-driven, not guessed.
        "Lasso Regression": LassoCV(
            alphas=100,        # generates 100 candidates on the reg path
            cv=5,
            max_iter=50_000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=6,
            min_samples_leaf=10,
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_leaf=5,
            max_features=0.7,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            min_samples_leaf=5,
        ),
        "Support Vector Regressor": SVR(
            kernel="rbf",
            C=5000,
            gamma="scale",
            epsilon=200,
        ),
        # ── Bonus ────────────────────────────────────────────────────────────
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_leaf=5,
            max_features=0.7,
            n_jobs=-1,
        ),
        "AdaBoost Regressor": AdaBoostRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            learning_rate=0.05,
        ),
    }

    # XGBoost — optional bonus model
    try:
        from xgboost import XGBRegressor

        registry["XGBoost Regressor"] = XGBRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            verbosity=0,
            n_jobs=-1,
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
    """
    scale = model_name in SCALED_MODELS
    preprocessor = build_preprocessor(
        scale_numeric=scale,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", estimator),
    ])


def _feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Extract transformed feature names from a fitted Pipeline."""
    ct: ColumnTransformer = pipeline.named_steps["preprocessor"]
    names: list[str] = []
    for tname, transformer, cols in ct.transformers_:
        if tname == "num":
            names.extend(cols)
        elif tname == "cat":
            enc = transformer.named_steps["encoder"]
            names.extend(enc.get_feature_names_out(cols).tolist())
    return names


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    list[str],
    list[str],
]:
    """
    Load → clean → engineer → split.
    """
    log.info("── Step 1/5  Loading raw data ──────────────────────────────")
    raw = load_raw_data()
    log.info("Raw data loaded: %s rows × %s cols", *raw.shape)

    log.info("── Step 2/5  Cleaning ──────────────────────────────────────")
    cleaned = clean_dataframe(raw)

    log.info("── Step 3/5  Feature engineering ───────────────────────────")
    enriched = engineer_features(cleaned)

    # Save enriched dataset for downstream use
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
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    num_feats: list[str],
    cat_feats: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """
    Train every model in the registry, evaluate, cross-validate.
    """
    registry = _get_model_registry()
    all_metrics: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    cv_results: dict[str, dict] = {}
    lc_data: dict[str, dict] = {}

    log.info("── Step 5/5  Training %d models ─────────────────────────────", len(registry))
    total = len(registry)

    for idx, (name, estimator) in enumerate(registry.items(), start=1):
        log.info("[%d/%d] %s", idx, total, name)

        pipeline = _build_pipeline(clone(estimator), num_feats, cat_feats, name)

        metrics, fitted, y_pred = evaluate_model(
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
            model_name=name,
            log_transformed=False,
        )
        all_metrics.append(metrics)
        fitted_models[name] = fitted
        predictions[name] = y_pred

        # Log CV-selected alpha for RidgeCV / LassoCV
        reg = fitted.named_steps["regressor"]
        if hasattr(reg, "alpha_"):
            log.info("  → CV-selected alpha = %.6g", reg.alpha_)

        # 5-fold cross-validation
        cv_pipeline = _build_pipeline(clone(estimator), num_feats, cat_feats, name)
        cv_results[name] = run_cross_validation(cv_pipeline, X_train, y_train)
        log.info(
            "  CV R² = %.4f ± %.4f",
            cv_results[name]["r2_mean"],
            cv_results[name]["r2_std"],
        )

        # Learning curve
        if name in {
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "Ridge Regression",
            "Lasso Regression",
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
    """
    ensure_output_dirs()

    bundle = {
        "model": fitted_models[best_name],
        "metadata": {
            "model_name": best_name,
            "target_column": TARGET_COLUMN,
            "numeric_features": num_feats,
            "categorical_features": cat_feats,
        },
    }

    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(bundle, model_path)
    log.info("Best model saved → %s", model_path)

    feature_info = {
        "best_model": best_name,
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
        "all_features": num_feats + cat_feats,
        "target_column": TARGET_COLUMN,
    }
    save_json(feature_info, MODELS_DIR / "feature_info.json")
    save_json(bundle["metadata"], MODELS_DIR / "model_metadata.json")

    return model_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_training_pipeline() -> dict[str, Any]:
    """
    Execute the complete training pipeline end-to-end and return a summary dict.
    """
    t_start = time.perf_counter()
    ensure_output_dirs()

    X_train, X_test, y_train, y_test, enriched, num_feats, cat_feats = prepare_data()

    metrics_df, fitted_models, predictions = train_all_models(
        X_train, X_test, y_train, y_test, num_feats, cat_feats
    )

    _cv_path = METRICS_DIR / "cross_validation.json"
    _cv_for_selection = json.loads(_cv_path.read_text()) if _cv_path.exists() else None
    best_name = select_best_model(metrics_df, cv_data=_cv_for_selection)
    generate_comparison_report(metrics_df, best_name)

    model_path = save_best_model(fitted_models, best_name, num_feats, cat_feats)

    log.info("Generating EDA figures...")
    generate_all_eda_figures(enriched)

    log.info("Generating evaluation figures...")
    lc_data_loaded: dict | None = None
    lc_path = METRICS_DIR / "learning_curves.json"
    if lc_path.exists():
        with lc_path.open() as _f:
            lc_data_loaded = json.load(_f)
    generate_all_evaluation_figures(
        metrics_df, fitted_models, predictions, y_test, lc_data_loaded
    )

    elapsed = time.perf_counter() - t_start
    log.info("\n%s", "=" * 65)
    log.info("  TRAINING COMPLETE  (%.1f s total)", elapsed)
    log.info("=" * 65)

    display = (
        metrics_df.sort_values("rmse")
        .assign(
            RMSE=lambda d: d["rmse"].apply(lambda v: f"${v:,.0f}"),
            MAE=lambda d: d["mae"].apply(lambda v: f"${v:,.0f}"),
            R2=lambda d: d["r2"].apply(lambda v: f"{v:.4f}"),
            Time=lambda d: d["train_time_sec"].apply(lambda v: f"{v:.2f}s"),
        )[["model", "RMSE", "MAE", "R2", "Time"]]
        .rename(columns={"model": "Model"})
        .reset_index(drop=True)
    )
    display.index += 1
    log.info("\n%s", display.to_string())
    log.info("\n  ⭐  Best model: %s", best_name)
    log.info("=" * 65 + "\n")

    summary = {
        "best_model": best_name,
        "model_path": str(model_path),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "elapsed_sec": round(elapsed, 2),
        "metrics": metrics_df.to_dict(orient="records"),
    }
    save_json(summary, REPORTS_DIR / "training_summary.json")
    return summary


if __name__ == "__main__":
    run_training_pipeline()
