"""
Inference module for Medical Insurance Cost prediction.

Loads the saved best_model.pkl and applies the same cleaning +
feature engineering pipeline that was used during training so
predictions are always consistent with training-time transformations.

Usage
-----
    from src.predict import predict_charges

    result = predict_charges({
        "age": 35, "sex": "male", "bmi": 28.5,
        "children": 2, "smoker": "no", "region": "northwest"
    })
    print(f"${result:,.2f}")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocessing import clean_dataframe
from src.utils import MODELS_DIR, TARGET_COLUMN, get_logger

log = get_logger("predict")

# Original raw feature columns (what the user provides at inference time)
RAW_INPUT_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]


# ---------------------------------------------------------------------------
from functools import lru_cache

# ---------------------------------------------------------------------------
# Model loading with LRU cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _load_model_cached(path_str: str) -> dict[str, Any]:
    model_path = Path(path_str)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run `python -m src.train` to train and save the model first."
        )
    bundle = joblib.load(model_path)
    log.info("Loaded model: %s", bundle["metadata"]["model_name"])
    return bundle


@lru_cache(maxsize=16)
def _load_feature_info_cached(path_str: str) -> dict[str, Any]:
    info_path = Path(path_str)
    if not info_path.exists():
        raise FileNotFoundError(
            f"Feature info not found at {info_path}.\n"
            "Run `python -m src.train` first."
        )
    with info_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model(path: Path | None = None) -> dict[str, Any]:
    """
    Load the saved model bundle from models/best_model.pkl (cached).
    """
    model_path = path or (MODELS_DIR / "best_model.pkl")
    return _load_model_cached(str(model_path.resolve()))


def load_feature_info(path: Path | None = None) -> dict[str, Any]:
    """Load feature metadata saved during training (cached)."""
    info_path = path or (MODELS_DIR / "feature_info.json")
    return _load_feature_info_cached(str(info_path.resolve()))


def clear_model_cache() -> None:
    """Clear in-memory cached model and feature info."""
    _load_model_cached.cache_clear()
    _load_feature_info_cached.cache_clear()


# ---------------------------------------------------------------------------
# Input preparation  (mirrors training pipeline, no data leakage)
# ---------------------------------------------------------------------------

def _prepare_single(raw: dict[str, Any], all_features: list[str]) -> pd.DataFrame:
    """
    Turn a raw input dict into a feature-engineered DataFrame row.
    """
    df = pd.DataFrame([raw])

    # Normalise strings — mirrors preprocessing.standardise_categoricals
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # Cast numerics
    for col in ["age", "bmi", "children"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply feature engineering (no target column present — safe)
    df = engineer_features(df)

    # Align to training feature list
    for col in all_features:
        if col not in df.columns:
            # Numeric engineered features default to 0; categoricals to "unknown"
            df[col] = 0 if col in ["is_obese", "smoker_obese",
                                    "age_bmi", "has_children", "age_smoker"] else "unknown"

    return df[all_features]


def _prepare_batch(df: pd.DataFrame, all_features: list[str]) -> pd.DataFrame:
    """Vectorised feature preparation for batch inference."""
    cleaned = df.copy()
    str_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        cleaned[col] = cleaned[col].astype(str).str.lower().str.strip()

    for col in ["age", "bmi", "children"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    engineered = engineer_features(cleaned)

    for col in all_features:
        if col not in engineered.columns:
            engineered[col] = 0 if col in ["is_obese", "smoker_obese",
                                           "age_bmi", "has_children", "age_smoker"] else "unknown"

    return engineered[all_features]


# ---------------------------------------------------------------------------
# Public prediction API
# ---------------------------------------------------------------------------

def predict_charges(
    raw_input: dict[str, Any],
    model_path: Path | None = None,
) -> float:
    """
    Predict insurance charges for a single policyholder.
    """
    bundle       = load_model(model_path)
    feature_info = load_feature_info()
    all_features = feature_info["all_features"]

    X = _prepare_single(raw_input, all_features)
    pred = bundle["model"].predict(X)[0]
    result = float(max(pred, 0.0))
    log.info("predict_charges → $%.2f", result)
    return result


def predict_batch(
    df: pd.DataFrame,
    model_path: Path | None = None,
) -> pd.Series:
    """
    Predict charges for a batch of raw policyholder records.
    """
    bundle       = load_model(model_path)
    feature_info = load_feature_info()
    all_features = feature_info["all_features"]

    X = _prepare_batch(df, all_features)

    preds = bundle["model"].predict(X)
    preds = np.clip(preds, 0, None)
    log.info("predict_batch: %d predictions returned.", len(preds))
    return pd.Series(preds, index=df.index, name="predicted_charges")


def get_model_info() -> dict[str, Any]:
    """
    Return model name, target column, and feature lists from saved metadata.
    """
    feature_info = load_feature_info()
    return {
        "model_name":           feature_info["best_model"],
        "target_column":        feature_info["target_column"],
        "numeric_features":     feature_info["numeric_features"],
        "categorical_features": feature_info["categorical_features"],
        "n_features_total":     len(feature_info["all_features"]),
    }

