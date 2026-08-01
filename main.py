"""
FastAPI — Medical Insurance Cost Prediction  (JSON API only)

All page rendering is handled by the Next.js frontend.
This server exposes only JSON endpoints + serves output images.

Endpoints
---------
  POST /api/predict              { age, sex, bmi, children, smoker, region }
  GET  /api/metrics              model comparison table
  GET  /api/cv                   5-fold cross-validation results
  GET  /api/dataset/summary      shape, target stats, smoker breakdown
  GET  /api/dataset/sample       first 20 rows as JSON
  GET  /api/feature-info         best model + feature lists
  GET  /api/eda-figures          list of available EDA figure URLs
  GET  /api/eval-figures/{stem}  per-model evaluation figure URLs
  GET  /outputs/...              static PNG files

Run
---
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.predict import predict_charges
from src.utils import (
    DATA_DIR, EVAL_DIR, EDA_DIR,
    METRICS_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COLUMN,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Insurance Cost Prediction API",
    description="REST API for the medical insurance ML pipeline.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Next.js dev server (3000) and prod (same origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output images (EDA + evaluation PNGs)
OUTPUTS_DIR = ROOT / "outputs"
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    age:      int   = Field(..., ge=0, le=120, examples=[35])
    sex:      str   = Field(..., examples=["male"])
    bmi:      float = Field(..., ge=10, le=80, examples=[28.5])
    children: int   = Field(..., ge=0, le=10, examples=[2])
    smoker:   str   = Field(..., examples=["no"])
    region:   str   = Field(..., examples=["northwest"])

    @field_validator("sex")
    @classmethod
    def _sex(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"male", "female"}:
            raise ValueError("sex must be 'male' or 'female'")
        return v

    @field_validator("smoker")
    @classmethod
    def _smoker(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"yes", "no"}:
            raise ValueError("smoker must be 'yes' or 'no'")
        return v

    @field_validator("region")
    @classmethod
    def _region(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"northeast", "northwest", "southeast", "southwest"}:
            raise ValueError("invalid region")
        return v


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def _raw() -> pd.DataFrame:
    if "raw" not in _cache:
        p = DATA_DIR / "insurance.csv"
        _cache["raw"] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return _cache["raw"]


def _metrics() -> pd.DataFrame:
    if "metrics" not in _cache:
        p = METRICS_DIR / "comparison.csv"
        _cache["metrics"] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return _cache["metrics"]


def _cv() -> dict:
    if "cv" not in _cache:
        p = METRICS_DIR / "cross_validation.json"
        _cache["cv"] = json.loads(p.read_text()) if p.exists() else {}
    return _cache["cv"]


def _fi() -> dict:
    if "fi" not in _cache:
        p = MODELS_DIR / "feature_info.json"
        _cache["fi"] = json.loads(p.read_text()) if p.exists() else {}
    return _cache["fi"]


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/api/predict")
async def predict(data: PredictRequest) -> JSONResponse:
    """Run the trained model and return predicted charges."""
    try:
        inp = data.model_dump()
        predicted = predict_charges(inp)

        raw = _raw()
        ctx: dict = {}
        if not raw.empty:
            ctx["overall_mean"]   = round(float(raw[TARGET_COLUMN].mean()), 2)
            ctx["smoker_mean"]    = round(float(raw[raw["smoker"] == "yes"][TARGET_COLUMN].mean()), 2)
            ctx["nonsmoker_mean"] = round(float(raw[raw["smoker"] == "no"][TARGET_COLUMN].mean()), 2)

        return JSONResponse({
            "predicted_charges": round(predicted, 2),
            "formatted":         f"${predicted:,.2f}",
            "model_used":        _fi().get("best_model", ""),
            "input":             inp,
            "context":           ctx,
            "risk_flags":        _risk_flags(inp),
        })
    except FileNotFoundError as exc:
        raise HTTPException(503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


@app.get("/api/metrics")
async def api_metrics() -> JSONResponse:
    """Sorted model comparison table."""
    df = _metrics()
    if df.empty:
        raise HTTPException(503, "Metrics not found — run training first.")
    return JSONResponse(
        df.sort_values("rmse")
          .assign(is_best=lambda d: d["model"] == _fi().get("best_model", ""))
          .to_dict(orient="records")
    )


@app.get("/api/cv")
async def api_cv() -> JSONResponse:
    """5-fold cross-validation results per model."""
    cv = _cv()
    if not cv:
        raise HTTPException(503, "CV data not found — run training first.")
    return JSONResponse(cv)


@app.get("/api/dataset/summary")
async def api_summary() -> JSONResponse:
    """Dataset shape, target statistics, and smoker/region breakdowns."""
    raw = _raw()
    if raw.empty:
        raise HTTPException(503, "Dataset not found.")
    tgt = raw[TARGET_COLUMN]
    q1, q3 = float(tgt.quantile(0.25)), float(tgt.quantile(0.75))
    iqr = q3 - q1

    smoker_grp = raw.groupby("smoker")[TARGET_COLUMN].agg(["mean", "median", "count"])
    region_grp = raw.groupby("region")[TARGET_COLUMN].agg(["mean", "median"]).sort_values("mean", ascending=False)

    return JSONResponse({
        "rows":    len(raw),
        "columns": int(len(raw.columns)),
        "missing": int(raw.isna().sum().sum()),
        "duplicates": int(raw.duplicated().sum()),
        "target": {
            "min":      round(float(tgt.min()), 2),
            "max":      round(float(tgt.max()), 2),
            "mean":     round(float(tgt.mean()), 2),
            "median":   round(float(tgt.median()), 2),
            "std":      round(float(tgt.std()), 2),
            "skew":     round(float(tgt.skew()), 4),
            "q1":       round(q1, 2),
            "q3":       round(q3, 2),
            "outliers": int(((tgt < q1 - 1.5 * iqr) | (tgt > q3 + 1.5 * iqr)).sum()),
        },
        "smoker": {
            s: {
                "mean":   round(float(smoker_grp.loc[s, "mean"]), 2),
                "median": round(float(smoker_grp.loc[s, "median"]), 2),
                "count":  int(smoker_grp.loc[s, "count"]),
            }
            for s in ["yes", "no"] if s in smoker_grp.index
        },
        "regions": [
            {
                "name":   r,
                "mean":   round(float(region_grp.loc[r, "mean"]), 2),
                "median": round(float(region_grp.loc[r, "median"]), 2),
            }
            for r in region_grp.index
        ],
    })


@app.get("/api/dataset/sample")
async def api_sample() -> JSONResponse:
    """First 20 rows as JSON records."""
    raw = _raw()
    if raw.empty:
        raise HTTPException(503, "Dataset not found.")
    return JSONResponse(raw.head(20).to_dict(orient="records"))


@app.get("/api/feature-info")
async def api_feature_info() -> JSONResponse:
    """Feature lists and best model name."""
    fi = _fi()
    if not fi:
        raise HTTPException(503, "Feature info not found — run training first.")
    return JSONResponse(fi)


@app.get("/api/eda-figures")
async def api_eda_figures() -> JSONResponse:
    """List of available EDA figure names and their /outputs URLs."""
    mapping = {
        "charges_distribution":    "Charges Distribution",
        "charges_by_smoker":       "Charges by Smoker Status",
        "bmi_vs_charges":          "BMI vs Charges",
        "age_vs_charges":          "Age vs Charges",
        "charges_by_region":       "Charges by Region",
        "charges_by_bmi_category": "Charges by BMI Category",
        "age_distribution":        "Age Distribution",
        "bmi_distribution":        "BMI Distribution",
        "children_vs_charges":     "Children vs Charges",
        "sex_distribution":        "Sex Distribution",
        "correlation_heatmap":     "Correlation Heatmap",
        "pairplot":                "Pairplot",
    }
    result = [
        {"stem": s, "label": l, "url": f"http://localhost:8000/outputs/figures/eda/{s}.png"}
        for s, l in mapping.items()
        if (EDA_DIR / f"{s}.png").exists()
    ]
    return JSONResponse(result)


@app.get("/api/eval-figures/{model_stem}")
async def api_eval_figures(model_stem: str) -> JSONResponse:
    """URLs for per-model evaluation plots (residuals, pred-vs-actual, etc.)."""
    def url(f: str) -> str | None:
        p = EVAL_DIR / f
        return f"http://localhost:8000/outputs/figures/evaluation/{f}" if p.exists() else None

    return JSONResponse({
        "residuals":      url(f"residuals_{model_stem}.png"),
        "pred_vs_actual": url(f"pred_vs_actual_{model_stem}.png"),
        "feature_imp":    url(f"feature_importance_{model_stem}.png"),
        "coefficients":   url(f"coefficients_{model_stem}.png"),
        "learning_curve": url(f"learning_curve_{model_stem}.png"),
    })


@app.get("/api/eval-models")
async def api_eval_models() -> JSONResponse:
    """List of model stems that have residual plots."""
    stems = [
        {"stem": p.stem.replace("residuals_", ""),
         "label": p.stem.replace("residuals_", "").replace("_", " ").title()}
        for p in sorted(EVAL_DIR.glob("residuals_*.png"))
    ]
    return JSONResponse(stems)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_flags(inp: dict) -> list[str]:
    flags = []
    if inp.get("smoker") == "yes":
        flags.append("Smoker — strongest cost driver (3.8× average premium)")
    if inp.get("bmi", 0) >= 30 and inp.get("smoker") == "yes":
        flags.append("Obese smoker — highest-risk combination")
    elif inp.get("bmi", 0) >= 30:
        flags.append("BMI ≥ 30 — elevated baseline risk")
    if inp.get("age", 0) >= 50:
        flags.append("Age ≥ 50 — charges increase sharply after 50")
    return flags
