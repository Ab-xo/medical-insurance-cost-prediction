"""Foundation module — paths, constants, logging, and I/O helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EDA_DIR     = FIGURES_DIR / "eda"
EVAL_DIR    = FIGURES_DIR / "evaluation"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "insurance.csv"

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_COLUMN = "charges"
RANDOM_STATE  = 42

NUMERIC_FEATURES     = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SMOKER_SURCHARGE_FACTOR = 3.0
BMI_OBESE_THRESHOLD     = 30.0


# ── Directory setup ───────────────────────────────────────────────────────────

def ensure_output_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in (FIGURES_DIR, EDA_DIR, EVAL_DIR, METRICS_DIR, REPORTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Safe to call multiple times."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load insurance.csv. Raises FileNotFoundError with a helpful message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from Kaggle and place it at data/insurance.csv"
        )
    return pd.read_csv(path)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Write data as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def save_markdown(content: str, path: Path) -> None:
    """Write a markdown string to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ── Dataset inspection ────────────────────────────────────────────────────────

def _section(title: str, width: int = 60) -> str:
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"


def inspect_dataframe(df: pd.DataFrame, logger: logging.Logger | None = None) -> dict[str, Any]:
    """Print a structured overview of df and return the same data as a dict."""
    log = logger or get_logger("inspect")
    summary: dict[str, Any] = {}

    log.info(_section("SHAPE"))
    log.info("Rows: %d  |  Columns: %d", *df.shape)
    summary["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    log.info(_section("COLUMNS & DTYPES"))
    dtype_map = {col: str(dt) for col, dt in df.dtypes.items()}
    for col, dt in dtype_map.items():
        log.info("  %-12s  %s", col, dt)
    summary["dtypes"] = dtype_map

    log.info(_section("MISSING VALUES"))
    missing_counts = df.isna().sum()
    missing_pct    = (df.isna().mean() * 100).round(2)
    if missing_counts.sum() == 0:
        log.info("  No missing values found.")
    else:
        for col in missing_counts[missing_counts > 0].index:
            log.info("  %-12s  %d missing  (%.1f%%)", col, missing_counts[col], missing_pct[col])
    summary["missing_values"] = missing_counts.astype(int).to_dict()
    summary["missing_pct"]    = missing_pct.to_dict()

    log.info(_section("DUPLICATE ROWS"))
    n_dupes = int(df.duplicated().sum())
    log.info("  %d duplicate row(s)  (%.2f%%)", n_dupes, n_dupes / len(df) * 100)
    summary["duplicate_rows"] = n_dupes

    log.info(_section("NUMERIC STATISTICS"))
    num_df = df.select_dtypes(include=[np.number])
    log.info("\n%s", num_df.describe().T.to_string())
    summary["numeric_stats"] = num_df.describe().T.to_dict()

    log.info(_section("CATEGORICAL VALUE COUNTS"))
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    summary["categorical_counts"] = {}
    for col in cat_cols:
        vc = df[col].value_counts()
        log.info("\n  %s:\n%s", col, vc.to_string())
        summary["categorical_counts"][col] = vc.astype(int).to_dict()

    if TARGET_COLUMN in df.columns:
        log.info(_section(f"TARGET: {TARGET_COLUMN}"))
        tgt = df[TARGET_COLUMN]
        stats = {
            "min":    float(tgt.min()),
            "max":    float(tgt.max()),
            "mean":   float(tgt.mean()),
            "median": float(tgt.median()),
            "std":    float(tgt.std()),
            "skew":   float(tgt.skew()),
        }
        for k, v in stats.items():
            log.info("  %-8s  %.4f", k, v)
        summary["target_stats"] = stats

        q1, q3 = float(tgt.quantile(0.25)), float(tgt.quantile(0.75))
        iqr = q3 - q1
        log.info("  Q1=%.2f  Q3=%.2f  IQR=%.2f", q1, q3, iqr)
        summary["target_quartiles"] = {"q1": q1, "q3": q3, "iqr": iqr}

        n_outliers = int(((tgt < q1 - 1.5 * iqr) | (tgt > q3 + 1.5 * iqr)).sum())
        log.info("  Outliers (1.5×IQR): %d  (%.1f%%)", n_outliers, n_outliers / len(tgt) * 100)
        summary["target_outliers_iqr"] = n_outliers

    if "smoker" in df.columns and TARGET_COLUMN in df.columns:
        log.info(_section("SMOKER vs NON-SMOKER"))
        grp = df.groupby("smoker")[TARGET_COLUMN].agg(["mean", "median", "count"])
        log.info("\n%s", grp.to_string())
        summary["smoker_charges"] = grp.to_dict()

    log.info("\n" + "=" * 60 + "\n  Inspection complete.\n" + "=" * 60 + "\n")
    return summary


def generate_data_understanding_report(df: pd.DataFrame) -> str:
    """Run inspection and save results as JSON + Markdown to outputs/reports/."""
    ensure_output_dirs()
    log = get_logger("data_understanding")
    summary = inspect_dataframe(df, logger=log)

    save_json(summary, REPORTS_DIR / "data_understanding.json")

    lines = [
        "# Data Understanding Report — Medical Insurance Costs",
        "",
        "## Shape",
        f"- **Rows:** {summary['shape']['rows']:,}",
        f"- **Columns:** {summary['shape']['columns']}",
        "",
        "## Columns & Types",
        "",
        "| Column | Type |",
        "|--------|------|",
    ]
    for col, dt in summary["dtypes"].items():
        lines.append(f"| `{col}` | `{dt}` |")

    lines += ["", "## Missing Values", ""]
    if sum(summary["missing_values"].values()) == 0:
        lines.append("No missing values found.")
    else:
        lines += ["| Column | Missing | % |", "|--------|---------|---|"]
        for col, cnt in summary["missing_values"].items():
            lines.append(f"| `{col}` | {cnt} | {summary['missing_pct'][col]}% |")

    lines += ["", f"## Duplicates", f"- **Count:** {summary['duplicate_rows']}", "", "## Target (`charges`)", ""]

    if "target_stats" in summary:
        ts = summary["target_stats"]
        tq = summary["target_quartiles"]
        lines += [
            "| Stat | Value |", "|------|-------|",
            f"| Min | ${ts['min']:,.2f} |",
            f"| Max | ${ts['max']:,.2f} |",
            f"| Mean | ${ts['mean']:,.2f} |",
            f"| Median | ${ts['median']:,.2f} |",
            f"| Std | ${ts['std']:,.2f} |",
            f"| Skewness | {ts['skew']:.4f} |",
            f"| Q1 | ${tq['q1']:,.2f} |",
            f"| Q3 | ${tq['q3']:,.2f} |",
            f"| IQR | ${tq['iqr']:,.2f} |",
            f"| Outliers (1.5×IQR) | {summary['target_outliers_iqr']} |",
        ]

    if "smoker_charges" in summary:
        lines += ["", "## Smoker vs Non-Smoker", "", "| Smoker | Mean | Median | Count |", "|--------|------|--------|-------|"]
        means   = summary["smoker_charges"]["mean"]
        medians = summary["smoker_charges"]["median"]
        counts  = summary["smoker_charges"]["count"]
        for key in means:
            lines.append(f"| {key} | ${means[key]:,.2f} | ${medians[key]:,.2f} | {int(counts[key])} |")

    lines += ["", "---", "_Generated by `src/utils.py`_"]
    content = "\n".join(lines)
    save_markdown(content, REPORTS_DIR / "data_understanding.md")
    log.info("Reports saved → outputs/reports/data_understanding.{json,md}")
    return content
