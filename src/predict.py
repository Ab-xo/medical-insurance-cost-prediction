"""Inference — load the saved model and predict insurance charges.

Usage:
    from src.predict import predict_charges

    charge = predict_charges({
        "age": 35, "sex": "male", "bmi": 28.5,
        "children": 2, "smoker": "no", "region": "northwest"
    })
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import engineer_features
from src.utils import MODELS_DIR, get_logger

log = get_logger("predict")

RAW_INPUT_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]

# Engineered numeric columns — default to 0 if missing at inference time
_NUMERIC_ENGINEERED = {"is_obese", "smoker_obese", "age_bmi", "has_children", "age_smoker"}


# ── Model loading (cached) ────────────────────────────────────────────────────

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
    """Load the model bundle from models/best_model.pkl (cached after first load)."""
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


# ── Input preparation ─────────────────────────────────────────────────────────

def _normalise_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip all string columns — mirrors training preprocessing."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
    return df


def _cast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce age, bmi, children to numeric."""
    for col in ["age", "bmi", "children"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _align_to_training(df: pd.DataFrame, all_features: list[str]) -> pd.DataFrame:
    """Add any missing columns with sensible defaults and select training columns."""
    for col in all_features:
        if col not in df.columns:
            df[col] = 0 if col in _NUMERIC_ENGINEERED else "unknown"
    return df[all_features]


def _prepare_single(raw: dict[str, Any], all_features: list[str]) -> pd.DataFrame:
    """Turn a single raw input dict into a model-ready DataFrame row."""
    df = pd.DataFrame([raw])
    df = _normalise_strings(df)
    df = _cast_numerics(df)
    df = engineer_features(df)
    return _align_to_training(df, all_features)


def _prepare_batch(df: pd.DataFrame, all_features: list[str]) -> pd.DataFrame:
    """Prepare a batch DataFrame for inference."""
    out = df.copy()
    out = _normalise_strings(out)
    out = _cast_numerics(out)
    out = engineer_features(out)
    return _align_to_training(out, all_features)


# ── Public API ────────────────────────────────────────────────────────────────

def predict_charges(raw_input: dict[str, Any], model_path: Path | None = None) -> float:
    """Predict annual insurance charges for a single policyholder."""
    bundle      = load_model(model_path)
    all_features = load_feature_info()["all_features"]

    X    = _prepare_single(raw_input, all_features)
    pred = float(max(bundle["model"].predict(X)[0], 0.0))
    log.info("predict_charges → $%.2f", pred)
    return pred


def predict_batch(df: pd.DataFrame, model_path: Path | None = None) -> pd.Series:
    """Predict charges for a batch of raw policyholder records."""
    bundle       = load_model(model_path)
    all_features = load_feature_info()["all_features"]

    X    = _prepare_batch(df, all_features)
    preds = np.clip(bundle["model"].predict(X), 0, None)
    log.info("predict_batch: %d predictions returned.", len(preds))
    return pd.Series(preds, index=df.index, name="predicted_charges")


def get_model_info() -> dict[str, Any]:
    """Return model name, target column, and feature lists from saved metadata."""
    fi = load_feature_info()
    return {
        "model_name":           fi["best_model"],
        "target_column":        fi["target_column"],
        "numeric_features":     fi["numeric_features"],
        "categorical_features": fi["categorical_features"],
        "n_features_total":     len(fi["all_features"]),
    }
