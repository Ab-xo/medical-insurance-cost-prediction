"""Model evaluation, cross-validation, and best-model selection.

All metrics are reported in USD (original scale), even when models were
trained on log-transformed targets.
"""

from __future__ import annotations

import json as _json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve

from src.utils import (
    METRICS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_output_dirs,
    get_logger,
    save_markdown,
)

log = get_logger("evaluate")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float = 0.0,
    predict_time: float = 0.0,
) -> dict[str, float]:
    """Return MAE, MSE, RMSE, R², and timing on the dollar scale."""
    return {
        "mae":              float(mean_absolute_error(y_true, y_pred)),
        "mse":              float(mean_squared_error(y_true, y_pred)),
        "rmse":             float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":               float(r2_score(y_true, y_pred)),
        "train_time_sec":   train_time,
        "predict_time_sec": predict_time,
    }


def evaluate_model(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    log_transformed: bool = False,
) -> tuple[dict[str, float], Any, np.ndarray]:
    """Fit pipeline, predict on test set, and return (metrics, fitted_pipeline, predictions).

    Set log_transformed=True if targets were log1p-encoded — predictions
    will be inverse-transformed before metrics are computed.
    """
    t0 = time.perf_counter()
    fitted = pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw_pred = fitted.predict(X_test)
    predict_time = time.perf_counter() - t1

    if log_transformed:
        y_true_usd = np.expm1(y_test.values)
        y_pred_usd = np.expm1(np.clip(raw_pred, 0, None))
    else:
        y_true_usd = y_test.values
        y_pred_usd = raw_pred

    metrics = compute_metrics(y_true_usd, y_pred_usd, train_time, predict_time)
    metrics["model"] = model_name

    log.info("%-32s  RMSE=$%s  MAE=$%s  R²=%.4f  t=%.2fs",
             model_name,
             f"{metrics['rmse']:>9,.0f}",
             f"{metrics['mae']:>8,.0f}",
             metrics["r2"],
             train_time)

    return metrics, fitted, y_pred_usd


def run_cross_validation(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run k-fold CV and return mean/std R² and RMSE across folds."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    r2_scores   = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", n_jobs=-1)
    rmse_scores = cross_val_score(pipeline, X, y, cv=cv,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)

    return {
        "n_splits":   n_splits,
        "r2_mean":    float(r2_scores.mean()),
        "r2_std":     float(r2_scores.std()),
        "r2_folds":   r2_scores.tolist(),
        "rmse_mean":  float(-rmse_scores.mean()),
        "rmse_std":   float(rmse_scores.std()),
        "rmse_folds": (-rmse_scores).tolist(),
    }


def compute_learning_curve(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    train_sizes: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Compute train vs validation R² at increasing training set sizes."""
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


def select_best_model(
    metrics_df: pd.DataFrame,
    cv_data: dict | None = None,
) -> str:
    """Pick the best model using a composite rank score.

    Weights: RMSE 45% · MAE 35% · R² 20%
    When models tie within 0.5 rank points, CV R² breaks the tie.
    """
    df = metrics_df.copy()

    df["rmse_rank"] = df["rmse"].rank(ascending=True)
    df["mae_rank"]  = df["mae"].rank(ascending=True)
    df["r2_rank"]   = df["r2"].rank(ascending=False)
    df["composite"] = 0.45 * df["rmse_rank"] + 0.35 * df["mae_rank"] + 0.20 * df["r2_rank"]

    best_score = df["composite"].min()
    candidates = df[df["composite"] <= best_score + 0.5].copy()

    if len(candidates) > 1 and cv_data is None:
        cv_path = METRICS_DIR / "cross_validation.json"
        if cv_path.exists():
            cv_data = _json.loads(cv_path.read_text())

    if len(candidates) > 1 and cv_data:
        candidates["cv_r2"] = candidates["model"].map(
            lambda m: cv_data.get(m, {}).get("r2_mean", 0.0)
        )
        best = candidates.loc[candidates["cv_r2"].idxmax(), "model"]
        log.info("select_best_model: tie broken by CV R²; winner = '%s'", best)
    else:
        best = df.loc[df["composite"].idxmin(), "model"]
        log.info("select_best_model: winner = '%s'", best)

    ranking = df.sort_values("composite")[["model", "rmse", "mae", "r2", "composite"]].reset_index(drop=True)
    log.info("Model ranking:\n%s", ranking.to_string(
        index=False,
        formatters={
            "rmse":      lambda v: f"${v:>9,.0f}",
            "mae":       lambda v: f"${v:>8,.0f}",
            "r2":        lambda v: f"{v:.4f}",
            "composite": lambda v: f"{v:.2f}",
        },
    ))
    return str(best)


def save_comparison_csv(metrics_df: pd.DataFrame) -> None:
    """Write the model comparison table to outputs/metrics/comparison.csv."""
    ensure_output_dirs()
    path = METRICS_DIR / "comparison.csv"
    metrics_df.to_csv(path, index=False)
    log.info("Saved comparison CSV → %s", path)


def generate_comparison_report(metrics_df: pd.DataFrame, best_model: str) -> str:
    """Build and save a Markdown leaderboard to outputs/reports/model_comparison.md."""
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
        "- RMSE 45%  MAE 35%  R² 20%",
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
        f"**{best_model}** — RMSE=${best_row['rmse']:,.0f}  MAE=${best_row['mae']:,.0f}  R²={best_row['r2']:.4f}",
        "",
        "---",
        "_Generated by `src/evaluate.py`_",
    ]

    content = "\n".join(lines)
    save_markdown(content, REPORTS_DIR / "model_comparison.md")
    log.info("Saved model comparison report → outputs/reports/model_comparison.md")
    return content
