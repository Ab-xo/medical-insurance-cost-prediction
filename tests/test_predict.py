"""
Unit tests for src/predict.py module.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.predict import (
    clear_model_cache,
    get_model_info,
    load_feature_info,
    load_model,
    predict_batch,
    predict_charges,
)


def test_load_model_cached():
    """Verify load_model returns model bundle and caches subsequent calls."""
    bundle1 = load_model()
    bundle2 = load_model()
    assert bundle1 is bundle2, "Cached load_model should return exact same object reference"
    assert "model" in bundle1
    assert "metadata" in bundle1


def test_load_feature_info_cached():
    """Verify load_feature_info returns metadata dict."""
    info1 = load_feature_info()
    info2 = load_feature_info()
    assert info1 is info2
    assert "all_features" in info1
    assert "best_model" in info1


def test_predict_charges_single():
    """Test single prediction on valid sample input."""
    raw_input = {
        "age": 35,
        "sex": "male",
        "bmi": 28.5,
        "children": 2,
        "smoker": "no",
        "region": "northwest",
    }
    pred = predict_charges(raw_input)
    assert isinstance(pred, float)
    assert pred > 0.0


def test_predict_batch_multiple():
    """Test batch prediction on DataFrame."""
    df = pd.DataFrame([
        {
            "age": 25,
            "sex": "female",
            "bmi": 22.0,
            "children": 0,
            "smoker": "no",
            "region": "southwest",
        },
        {
            "age": 55,
            "sex": "male",
            "bmi": 34.0,
            "children": 1,
            "smoker": "yes",
            "region": "southeast",
        },
    ])
    preds = predict_batch(df)
    assert len(preds) == 2
    assert isinstance(preds, pd.Series)
    assert preds.iloc[0] > 0.0
    # Smoker with higher age & BMI should have higher predicted cost
    assert preds.iloc[1] > preds.iloc[0]


def test_get_model_info():
    """Test get_model_info returns expected structure."""
    info = get_model_info()
    assert "model_name" in info
    assert "target_column" in info
    assert "numeric_features" in info
    assert "categorical_features" in info
    assert info["target_column"] == "charges"


def test_clear_model_cache():
    """Test clear_model_cache invalidates LRU cache."""
    bundle1 = load_model()
    clear_model_cache()
    bundle2 = load_model()
    assert bundle1 is not bundle2
