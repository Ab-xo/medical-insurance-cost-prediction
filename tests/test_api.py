"""
Integration tests for FastAPI REST API endpoints in main.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_api_predict_success():
    """Test POST /api/predict with valid payload."""
    payload = {
        "age": 30,
        "sex": "male",
        "bmi": 25.0,
        "children": 1,
        "smoker": "no",
        "region": "northeast",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_charges" in data
    assert "formatted" in data
    assert "model_used" in data
    assert "context" in data
    assert data["predicted_charges"] > 0


def test_api_predict_invalid_input():
    """Test POST /api/predict with invalid age/smoker validation."""
    payload = {
        "age": -5,  # Out of bounds
        "sex": "male",
        "bmi": 25.0,
        "children": 1,
        "smoker": "invalid_smoker_val",
        "region": "northeast",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_api_metrics():
    """Test GET /api/metrics endpoint."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "model" in data[0]
    assert "rmse" in data[0]
    assert "mae" in data[0]
    assert "r2" in data[0]


def test_api_cv():
    """Test GET /api/cv endpoint."""
    response = client.get("/api/cv")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_api_dataset_summary():
    """Test GET /api/dataset/summary endpoint."""
    response = client.get("/api/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert "rows" in data
    assert "target" in data
    assert data["rows"] > 0


def test_api_dataset_sample():
    """Test GET /api/dataset/sample endpoint."""
    response = client.get("/api/dataset/sample")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 20


def test_api_feature_info():
    """Test GET /api/feature-info endpoint."""
    response = client.get("/api/feature-info")
    assert response.status_code == 200
    data = response.json()
    assert "best_model" in data
    assert "all_features" in data


def test_api_eda_figures():
    """Test GET /api/eda-figures endpoint."""
    response = client.get("/api/eda-figures")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
