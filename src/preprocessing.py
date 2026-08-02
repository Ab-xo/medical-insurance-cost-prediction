"""Data cleaning, sklearn preprocessing pipeline, and train/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    get_logger,
)

log = get_logger("preprocessing")

# Valid ranges per column — values outside these are set to NaN
VALID_BOUNDS: dict[str, tuple[float, float]] = {
    "age":      (0,   120),
    "bmi":      (10,  80),
    "children": (0,   10),
    "charges":  (0,   1_000_000),
}

# Expected category values — anything else triggers a warning
VALID_CATEGORIES: dict[str, set[str]] = {
    "sex":    {"male", "female"},
    "smoker": {"yes", "no"},
    "region": {"northeast", "northwest", "southeast", "southwest"},
}


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(cleaned)
    if dropped:
        log.info("remove_duplicates: dropped %d duplicate row(s). %d → %d rows.", dropped, before, len(cleaned))
    else:
        log.info("remove_duplicates: no duplicates found.")
    return cleaned


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Set out-of-range numeric values to NaN so the imputer handles them."""
    cleaned = df.copy()
    for col, (lo, hi) in VALID_BOUNDS.items():
        if col not in cleaned.columns:
            continue
        mask = (cleaned[col] < lo) | (cleaned[col] > hi)
        n_invalid = int(mask.sum())
        if n_invalid:
            log.warning("validate_ranges: %d invalid value(s) in '%s' → NaN.", n_invalid, col)
            cleaned.loc[mask, col] = np.nan
        else:
            log.info("validate_ranges: '%s' — all values within [%g, %g].", col, lo, hi)
    return cleaned


def standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip all string columns. Warn on unexpected category values."""
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = (
            cleaned[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"nan": np.nan, "none": np.nan, "": np.nan})
        )

    for col, valid_set in VALID_CATEGORIES.items():
        if col not in cleaned.columns:
            continue
        unexpected = set(cleaned[col].dropna().unique()) - valid_set
        if unexpected:
            log.warning("standardise_categoricals: unexpected values in '%s': %s", col, unexpected)
        else:
            log.info("standardise_categoricals: '%s' — all values valid.", col)

    return cleaned


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values — median for numeric, mode for categorical.
    Drops rows where the target is missing."""
    cleaned = df.copy()

    if TARGET_COLUMN in cleaned.columns:
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
        if len(cleaned) < before:
            log.warning("handle_missing: dropped %d rows with missing target.", before - len(cleaned))

    num_cols = [c for c in cleaned.select_dtypes(include=[np.number]).columns if c != TARGET_COLUMN]
    for col in num_cols:
        n_null = int(cleaned[col].isna().sum())
        if n_null:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
            log.info("handle_missing: '%s' — filled %d NaN with median %.4f.", col, n_null, median_val)

    for col in cleaned.select_dtypes(include=["object", "string", "category"]).columns:
        n_null = int(cleaned[col].isna().sum())
        if n_null:
            mode_val = cleaned[col].mode()[0]
            cleaned[col] = cleaned[col].fillna(mode_val)
            log.info("handle_missing: '%s' — filled %d NaN with mode '%s'.", col, n_null, mode_val)

    return cleaned


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning sequence: dedup → normalise → validate → impute."""
    log.info("=== Starting clean_dataframe ===  input shape: %s", df.shape)
    df = remove_duplicates(df)
    df = standardise_categoricals(df)
    df = validate_ranges(df)
    df = handle_missing(df)
    df = df.reset_index(drop=True)
    log.info("=== clean_dataframe complete ===  output shape: %s", df.shape)
    return df


def build_preprocessor(
    scale_numeric: bool = True,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer with separate numeric and categorical pipelines.

    scale_numeric=True  → adds StandardScaler (required for linear models and SVR).
    scale_numeric=False → skips scaling (preferred for tree-based models).
    """
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    numeric_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",            # avoids the dummy variable trap
            handle_unknown="ignore", # safe when new values appear at inference
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Return feature names from a fitted ColumnTransformer in column order."""
    names: list[str] = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "num":
            names.extend(columns)
        elif transformer_name == "cat":
            encoder: OneHotEncoder = transformer.named_steps["encoder"]
            names.extend(encoder.get_feature_names_out(columns).tolist())
    return names


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    log_transform_target: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split.

    Stratification key = smoker_status + charge_quartile so both the smoker
    ratio (~20%) and the charge distribution are preserved in each split.
    """
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    if log_transform_target:
        y = np.log1p(y)
        log.info("split_data: applied log1p transform to target.")

    charge_quartile = pd.qcut(df[TARGET_COLUMN], q=4, labels=False)
    strat_key = df["smoker"].astype(str) + "_q" + charge_quartile.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=strat_key,
    )

    log.info("split_data: train=%d rows  test=%d rows  (%.0f / %.0f split)",
             len(X_train), len(X_test), (1 - test_size) * 100, test_size * 100)

    if "smoker" in X_train.columns:
        log.info("split_data: smoker %% — train=%.1f%%  test=%.1f%%",
                 (X_train["smoker"] == "yes").mean() * 100,
                 (X_test["smoker"]  == "yes").mean() * 100)

    return X_train, X_test, y_train, y_test
