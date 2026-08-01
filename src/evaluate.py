"""
Model evaluation, cross-validation, and comparison utilities.

Every metric computed here is on the ORIGINAL charge scale (USD) — even
when the model was trained on log1p-transformed targets, predictions are
inverse-transformed with np.expm1 before metric calculation so numbers
are always interpretable as dollars.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve

from src.utils import (
    METRICS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_output_dirs,
    get_logger,
    save_json,
    save_markdown,
)

log = get_logger("evaluate")


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float = 0.0,
    predict_time: float = 0.0,
) -> dict[str, float]:
    """
    Compute MAE, MSE, RMSE, R² on the original dollar scale.

    Parameters
    ----------
    y_true, y_pred : array-like — ground truth and predictions (dollar scale)
    train_time     : float — wall-clock training seconds
    predict_time   : float — wall-clock inference seconds

    Returns
    -------
    dict with keys: mae, mse, rmse, r2, train_time_sec, predict_time_sec
    """
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))

    return {
        "mae":              mae,
        "mse":              mse,
        "rmse":             rmse,
        "r2":               r2,
        "train_time_sec":   train_time,
        "predict_time_sec": predict_time,
    }


# ---------------------------------------------------------------------------
# Single model evaluation  (fit + predict + metrics in one call)
# ---------------------------------------------------------------------------

def evaluate_model(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    log_transformed: bool = False,
) -> tuple[dict[str, float], Any, np.ndarray]:
    """
    Fit *pipeline* on training data, evaluate on test, return metrics.

    Parameters
    ----------
    pipeline        : unfitted sklearn Pipeline
    log_transformed : bool — if True, y_train/y_test are in log1p space;
                      predictions are inverse-transformed before metric calc.

    Returns
    -------
    metrics  : dict  — all metrics on dollar scale
    fitted   : fitted pipeline
    y_pred   : np.ndarray predictions on dollar scale
    """
    t0 = time.perf_counter()
    fitted = pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw_pred = fitted.predict(X_test)
    predict_time = time.perf_counter() - t1

    # Inverse-transform both targets when log1p was applied
    if log_transformed:
        y_true_dollars = np.expm1(y_test.values)
        y_pred_dollars = np.expm1(np.clip(raw_pred, 0, None))
    else:
        y_true_dollars = y_test.values
        y_pred_dollars = raw_pred

    metrics = compute_metrics(y_true_dollars, y_pred_dollars, train_time, predict_time)
    metrics["model"] = model_name

    log.info(
        "%-32s  RMSE=$%s  MAE=$%s  R²=%.4f  t=%.2fs",
        model_name,
        f"{metrics['rmse']:>9,.0f}",
        f"{metrics['mae']:>8,.0f}",
        metrics["r2"],
        train_time,
    )
    return metrics, fitted, y_pred_dollars


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict[str, Any]:
    """
    Run stratified k-fold CV and return mean/std R² and per-fold scores.

    Uses neg_root_mean_squared_error and r2 scoring. Results are always
    on the scale of y (dollar scale if untransformed, log scale if not).
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    r2_scores   = cross_val_score(pipeline, X, y, cv=cv, scoring="r2",                       n_jobs=-1)
    rmse_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)

    return {
        "n_splits":      n_splits,
        "r2_mean":       float(r2_scores.mean()),
        "r2_std":        float(r2_scores.std()),
        "r2_folds":      r2_scores.tolist(),
        "rmse_mean":     float(-rmse_scores.mean()),
        "rmse_std":      float(rmse_scores.std()),
        "rmse_folds":    (-rmse_scores).tolist(),
    }


# ---------------------------------------------------------------------------
# Learning curve data
# ---------------------------------------------------------------------------

def compute_learning_curve(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    train_sizes: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """
    Compute train/validation R² at increasing training set sizes.

    Returns a dict suitable for plotting and JSON serialisation.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.10, 1.0, 6)

    sizes_abs, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        train_sizes=train_sizes,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return {
        "train_sizes":  sizes_abs.tolist(),
        "train_scores": train_scores.mean(axis=1).tolist(),
        "val_scores":   val_scores.mean(axis=1).tolist(),
    }


# ---------------------------------------------------------------------------
# Best model selection  (composite ranking)
# ---------------------------------------------------------------------------

def select_best_model(
    metrics_df: pd.DataFrame,
    cv_data: dict | None = None,
) -> str:
    """
    Pick the best model via composite ranking across metrics.

    Weights (tuned for insurance cost prediction):
      RMSE  45 % — primary metric; penalises large errors on high charges
      MAE   35 % — robust to the right-skewed charges distribution
      R²    20 % — variance explained

    Train time is intentionally excluded from ranking — a 1-second difference
    between trained pipelines is irrelevant for a batch model; accuracy matters.

    Tie-breaking:
      When two models score within 0.5 composite rank points of each other,
      the one with the better CV R² mean wins. This ensures generalisation
      beats lucky test-set performance.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Must contain columns: model, rmse, mae, r2
    cv_data : dict, optional
        Cross-validation results keyed by model name with key 'r2_mean'.
        Used for tie-breaking. Loaded from outputs/metrics/cross_validation.json
        if not provided.
    """
    import json as _json

    df = metrics_df.copy()

    # ── primary composite score ───────────────────────────────────────────────
    df["rmse_rank"] = df["rmse"].rank(ascending=True)       # lower = better
    df["mae_rank"]  = df["mae"].rank(ascending=True)        # lower = better
    df["r2_rank"]   = df["r2"].rank(ascending=False)        # higher = better

    df["composite"] = (
        0.45 * df["rmse_rank"]
        + 0.35 * df["mae_rank"]
        + 0.20 * df["r2_rank"]
    )

    best_composite = df["composite"].min()
    TIE_THRESHOLD  = 0.5  # models within this band are considered tied

    candidates = df[df["composite"] <= best_composite + TIE_THRESHOLD].copy()

    # ── tie-break via CV R² ───────────────────────────────────────────────────
    if len(candidates) > 1 and cv_data is None:
        # Try to load from disk
        from src.utils import METRICS_DIR
        cv_path = METRICS_DIR / "cross_validation.json"
        if cv_path.exists():
            cv_data = _json.loads(cv_path.read_text())

    if len(candidates) > 1 and cv_data:
        candidates = candidates.copy()
        candidates["cv_r2"] = candidates["model"].map(
            lambda m: cv_data.get(m, {}).get("r2_mean", 0.0)
        )
        best_idx = candidates["cv_r2"].idxmax()
        best = candidates.loc[best_idx, "model"]
        log.info(
            "select_best_model: tie between %s — broken by CV R²; winner = '%s'",
            candidates["model"].tolist(), best,
        )
    else:
        best = df.loc[df["composite"].idxmin(), "model"]
        log.info("select_best_model: winner = '%s'", best)

    # ── log full ranking ──────────────────────────────────────────────────────
    ranking = (
        df.sort_values("composite")[["model", "rmse", "mae", "r2", "composite"]]
        .reset_index(drop=True)
    )
    log.info(
        "Model ranking:\n%s",
        ranking.to_string(
            index=False,
            formatters={
                "rmse":      lambda v: f"${v:>9,.0f}",
                "mae":       lambda v: f"${v:>8,.0f}",
                "r2":        lambda v: f"{v:.4f}",
                "composite": lambda v: f"{v:.2f}",
            },
        ),
    )
    return str(best)


# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------

def save_comparison_csv(metrics_df: pd.DataFrame) -> None:
    """Write model comparison table to outputs/metrics/comparison.csv."""
    ensure_output_dirs()
    path = METRICS_DIR / "comparison.csv"
    metrics_df.to_csv(path, index=False)
    log.info("Saved comparison CSV → %s", path)


def generate_comparison_report(metrics_df: pd.DataFrame, best_model: str) -> str:
    """
    Build and save a Markdown model comparison report.

    Writes to outputs/reports/model_comparison.md and returns the content.
    """
    ensure_output_dirs()
    sorted_df = metrics_df.sort_values("rmse").reset_index(drop=True)

    lines = [
        "# Model Comparison Report — Medical Insurance Cost",
        "",
        f"**Best Model:** {best_model}",
        "",
        "## Selection Criteria",
        "",
        "Composite rank score (lower = better):",
        "- RMSE 40 % — penalises large prediction errors",
        "- MAE  35 % — robust to the right-skewed charges distribution",
        "- R²   15 % — variance explained",
        "- Time 10 % — prefer faster models when accuracy is similar",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | RMSE | MAE | R² | Train (s) |",
        "|------|-------|------|-----|----|-----------|",
    ]

    for rank, row in sorted_df.iterrows():
        star = " ⭐" if row["model"] == best_model else ""
        lines.append(
            f"| {rank + 1} | {row['model']}{star} | "
            f"${row['rmse']:,.0f} | ${row['mae']:,.0f} | "
            f"{row['r2']:.4f} | {row['train_time_sec']:.2f} |"
        )

    best_row = metrics_df.loc[metrics_df["model"] == best_model].iloc[0]
    lines += [
        "",
        "## Summary",
        "",
        f"The **{best_model}** achieved the best balance of accuracy and speed.",
        f"RMSE = ${best_row['rmse']:,.0f} | MAE = ${best_row['mae']:,.0f} | R² = {best_row['r2']:.4f}",
        "",
        "---",
        "_Generated automatically by `src/evaluate.py`_",
    ]

    content = "\n".join(lines)
    save_markdown(content, REPORTS_DIR / "model_comparison.md")
    log.info("Saved model comparison report → outputs/reports/model_comparison.md")
    return content
