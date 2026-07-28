"""
Utility functions for paths, I/O, logging, and dataset inspection.

This is the foundation module — every other src/ file imports from here.
Nothing in utils.py imports from other src/ modules (no circular deps).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project-level paths  (resolved from this file's location → project root)
# ---------------------------------------------------------------------------

ROOT_DIR    = Path(__file__).resolve().parents[1]   # medical-insurance-cost-prediction/
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EDA_DIR     = FIGURES_DIR / "eda"
EVAL_DIR    = FIGURES_DIR / "evaluation"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "insurance.csv"

# ---------------------------------------------------------------------------
# Dataset-level constants
# ---------------------------------------------------------------------------

TARGET_COLUMN = "charges"
RANDOM_STATE  = 42          # fixed seed — keeps results reproducible

# Feature groups (matches the insurance.csv schema)
NUMERIC_FEATURES     = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Domain knowledge
SMOKER_SURCHARGE_FACTOR = 3.0   # smokers pay ~3× more on average
BMI_OBESE_THRESHOLD     = 30.0  # BMI ≥ 30 is clinically obese


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create every output directory if it does not already exist."""
    for directory in (FIGURES_DIR, EDA_DIR, EVAL_DIR, METRICS_DIR, REPORTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with a console handler.

    Safe to call multiple times — handler is only added once.
    """
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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw insurance CSV.

    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from Kaggle and place it at data/insurance.csv"
        )
    df = pd.read_csv(path)
    return df


def save_json(data: dict[str, Any], path: Path) -> None:
    """Serialise *data* as indented JSON and write to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def save_markdown(content: str, path: Path) -> None:
    """Write a markdown string to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset inspection  (raw overview before ANY preprocessing)
# ---------------------------------------------------------------------------

def _section(title: str, width: int = 60) -> str:
    """Return a formatted section header string."""
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"


def inspect_dataframe(df: pd.DataFrame, logger: logging.Logger | None = None) -> dict[str, Any]:
    """
    Print and return a structured overview of *df* with no modifications.

    Covers:
      - Shape & column list
      - Data types
      - Missing values (count + %)
      - Duplicate rows
      - Numeric descriptive statistics
      - Categorical value counts
      - Target variable (charges) summary
      - Class balance of key categorical features

    Returns
    -------
    dict
        Same information as a nested dict (useful for saving to JSON).
    """
    log = logger or get_logger("inspect")
    summary: dict[str, Any] = {}

    # ---- Shape ----
    log.info(_section("SHAPE"))
    log.info("Rows: %d  |  Columns: %d", *df.shape)
    summary["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    # ---- Columns & dtypes ----
    log.info(_section("COLUMNS & DTYPES"))
    dtype_map = {col: str(dt) for col, dt in df.dtypes.items()}
    for col, dt in dtype_map.items():
        log.info("  %-12s  %s", col, dt)
    summary["dtypes"] = dtype_map

    # ---- Missing values ----
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

    # ---- Duplicates ----
    log.info(_section("DUPLICATE ROWS"))
    n_dupes = int(df.duplicated().sum())
    log.info("  %d duplicate row(s)  (%.2f%% of dataset)", n_dupes, n_dupes / len(df) * 100)
    summary["duplicate_rows"] = n_dupes

    # ---- Numeric descriptive stats ----
    log.info(_section("NUMERIC FEATURE STATISTICS"))
    num_df = df.select_dtypes(include=[np.number])
    log.info("\n%s", num_df.describe().T.to_string())
    summary["numeric_stats"] = num_df.describe().T.to_dict()

    # ---- Categorical value counts ----
    log.info(_section("CATEGORICAL FEATURE VALUE COUNTS"))
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    summary["categorical_counts"] = {}
    for col in cat_cols:
        vc = df[col].value_counts()
        log.info("\n  %s:\n%s", col, vc.to_string())
        summary["categorical_counts"][col] = vc.astype(int).to_dict()

    # ---- Target variable deep-dive ----
    if TARGET_COLUMN in df.columns:
        log.info(_section(f"TARGET VARIABLE: {TARGET_COLUMN}"))
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

        # Quartile breakdown
        q1, q3 = float(tgt.quantile(0.25)), float(tgt.quantile(0.75))
        log.info("  Q1 = %.2f  |  Q3 = %.2f  |  IQR = %.2f", q1, q3, q3 - q1)
        summary["target_quartiles"] = {"q1": q1, "q3": q3, "iqr": q3 - q1}

        # Outlier estimate via 1.5×IQR rule
        iqr = q3 - q1
        n_outliers = int(((tgt < q1 - 1.5 * iqr) | (tgt > q3 + 1.5 * iqr)).sum())
        log.info("  Potential outliers (1.5×IQR rule): %d  (%.1f%%)",
                 n_outliers, n_outliers / len(tgt) * 100)
        summary["target_outliers_iqr"] = n_outliers

    # ---- Smoker vs non-smoker charges preview ----
    if "smoker" in df.columns and TARGET_COLUMN in df.columns:
        log.info(_section("SMOKER vs NON-SMOKER CHARGES"))
        grp = df.groupby("smoker")[TARGET_COLUMN].agg(["mean", "median", "count"])
        log.info("\n%s", grp.to_string())
        summary["smoker_charges"] = grp.to_dict()

    log.info("\n" + "=" * 60)
    log.info("  Inspection complete.")
    log.info("=" * 60 + "\n")

    return summary


def generate_data_understanding_report(df: pd.DataFrame) -> str:
    """
    Run inspect_dataframe and persist results as both JSON and Markdown.

    Writes to:
      outputs/reports/data_understanding.json
      outputs/reports/data_understanding.md

    Returns the markdown string.
    """
    ensure_output_dirs()
    log = get_logger("data_understanding")
    summary = inspect_dataframe(df, logger=log)

    # Persist JSON
    save_json(summary, REPORTS_DIR / "data_understanding.json")

    # Build Markdown report
    lines = [
        "# Data Understanding Report — Medical Insurance Costs",
        "",
        "## Dataset Shape",
        f"- **Rows:** {summary['shape']['rows']:,}",
        f"- **Columns:** {summary['shape']['columns']}",
        "",
        "## Columns & Data Types",
        "",
        "| Column | Type |",
        "|--------|------|",
    ]
    for col, dt in summary["dtypes"].items():
        lines.append(f"| `{col}` | `{dt}` |")

    lines += [
        "",
        "## Missing Values",
        "",
    ]
    total_missing = sum(summary["missing_values"].values())
    if total_missing == 0:
        lines.append("No missing values found in the dataset.")
    else:
        lines += ["| Column | Missing | % |", "|--------|---------|---|"]
        for col, cnt in summary["missing_values"].items():
            pct = summary["missing_pct"][col]
            lines.append(f"| `{col}` | {cnt} | {pct}% |")

    lines += [
        "",
        f"## Duplicate Rows",
        f"- **Count:** {summary['duplicate_rows']}",
        "",
        "## Target Variable (`charges`)",
        "",
    ]
    if "target_stats" in summary:
        ts = summary["target_stats"]
        tq = summary["target_quartiles"]
        lines += [
            f"| Stat | Value |",
            f"|------|-------|",
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
        lines += [
            "",
            "## Smoker vs Non-Smoker Charges",
            "",
            "| Smoker | Mean | Median | Count |",
            "|--------|------|--------|-------|",
        ]
        means   = summary["smoker_charges"]["mean"]
        medians = summary["smoker_charges"]["median"]
        counts  = summary["smoker_charges"]["count"]
        for key in means:
            lines.append(
                f"| {key} | ${means[key]:,.2f} | ${medians[key]:,.2f} | {int(counts[key])} |"
            )

    lines += ["", "---", "_Generated automatically by `src/utils.py`_"]
    content = "\n".join(lines)
    save_markdown(content, REPORTS_DIR / "data_understanding.md")
    log.info("Reports saved → outputs/reports/data_understanding.{json,md}")
    return content
