"""
Data cleaning and preprocessing for the Medical Insurance Cost dataset.

Pipeline overview (in order):
  1. remove_duplicates        — drop the 1 known duplicate row
  2. validate_ranges          — nullify/flag any biologically impossible values
  3. standardise_categoricals — lowercase + strip all string columns
  4. handle_missing           — median-fill numerics, mode-fill categoricals
                                (dataset is clean now, but pipeline stays robust)
  5. build_preprocessor       — sklearn ColumnTransformer ready for train/test use
     • numeric  → SimpleImputer(median) + optional StandardScaler
     • categorical → SimpleImputer(most_frequent) + OneHotEncoder
  6. split_data               — stratified train/test split, keeps smoker ratio

Key design decisions driven by data_understanding.md:
  - No missing values found  → imputer steps are defensive guards only
  - 1 duplicate row          → removed in clean_dataframe()
  - Charges skew = 1.52      → log1p target transform available here
  - Outliers are real        → NOT removed (high-charge smokers are valid)
  - Categoricals are clean   → only need lowercase normalisation + OHE
"""

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

# ---------------------------------------------------------------------------
# Domain-level validity bounds  (biologically / actuarially sensible ranges)
# ---------------------------------------------------------------------------
VALID_BOUNDS: dict[str, tuple[float, float]] = {
    "age":      (0,   120),
    "bmi":      (10,  80),
    "children": (0,   10),
    "charges":  (0,   1_000_000),
}

VALID_CATEGORIES: dict[str, set[str]] = {
    "sex":    {"male", "female"},
    "smoker": {"yes", "no"},
    "region": {"northeast", "northwest", "southeast", "southwest"},
}


# ---------------------------------------------------------------------------
# Step 1 — Remove duplicates
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows and reset the index.

    The dataset has 1 known duplicate (found in data_understanding.md).
    """
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(cleaned)
    if dropped:
        log.info("remove_duplicates: dropped %d duplicate row(s). %d → %d rows.",
                 dropped, before, len(cleaned))
    else:
        log.info("remove_duplicates: no duplicates found.")
    return cleaned


# ---------------------------------------------------------------------------
# Step 2 — Validate ranges  (flag impossible values as NaN)
# ---------------------------------------------------------------------------

def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nullify numeric values that fall outside biologically valid bounds.

    This does NOT drop rows — it converts impossible values to NaN so
    the imputer in the sklearn pipeline can handle them gracefully.
    """
    cleaned = df.copy()
    for col, (lo, hi) in VALID_BOUNDS.items():
        if col not in cleaned.columns:
            continue
        mask = (cleaned[col] < lo) | (cleaned[col] > hi)
        n_invalid = int(mask.sum())
        if n_invalid:
            log.warning("validate_ranges: %d out-of-range value(s) in '%s' → set to NaN.",
                        n_invalid, col)
            cleaned.loc[mask, col] = np.nan
        else:
            log.info("validate_ranges: '%s' — all values within [%g, %g].", col, lo, hi)
    return cleaned


# ---------------------------------------------------------------------------
# Step 3 — Standardise categorical columns
# ---------------------------------------------------------------------------

def standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase and strip whitespace from all string/object columns.

    Also validates that category values belong to the expected sets and
    logs a warning (but does NOT drop) any unexpected values.
    """
    cleaned = df.copy()
    str_cols = cleaned.select_dtypes(include=["object", "string"]).columns

    for col in str_cols:
        cleaned[col] = (
            cleaned[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"nan": np.nan, "none": np.nan, "": np.nan})
        )

    # Validate known categorical domains
    for col, valid_set in VALID_CATEGORIES.items():
        if col not in cleaned.columns:
            continue
        unexpected = set(cleaned[col].dropna().unique()) - valid_set
        if unexpected:
            log.warning("standardise_categoricals: unexpected values in '%s': %s",
                        col, unexpected)
        else:
            log.info("standardise_categoricals: '%s' — all values valid.", col)

    return cleaned


# ---------------------------------------------------------------------------
# Step 4 — Handle missing values  (defensive; dataset is clean)
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute any remaining NaN values.

    Strategy:
      - Numeric  → median  (robust to the right-skewed charges distribution)
      - Categorical → mode  (most-frequent category)

    The dataset has no missing values post-cleaning, but this step makes
    the pipeline robust to future data quality issues.
    """
    cleaned = df.copy()

    # Drop rows with missing target — a model cannot train without a label
    if TARGET_COLUMN in cleaned.columns:
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
        dropped = before - len(cleaned)
        if dropped:
            log.warning("handle_missing: dropped %d rows with missing target.", dropped)

    # Numeric imputation (median)
    num_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET_COLUMN]
    for col in num_cols:
        n_null = int(cleaned[col].isna().sum())
        if n_null:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
            log.info("handle_missing: filled %d NaN in '%s' with median=%.4f.",
                     n_null, col, median_val)

    # Categorical imputation (mode)
    cat_cols = cleaned.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in cat_cols:
        n_null = int(cleaned[col].isna().sum())
        if n_null:
            mode_val = cleaned[col].mode()[0]
            cleaned[col] = cleaned[col].fillna(mode_val)
            log.info("handle_missing: filled %d NaN in '%s' with mode='%s'.",
                     n_null, col, mode_val)

    return cleaned


# ---------------------------------------------------------------------------
# Master cleaning pipeline
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning sequence on the raw dataframe.

    Steps (in order):
      1. remove_duplicates
      2. standardise_categoricals
      3. validate_ranges
      4. handle_missing

    Returns a clean, index-reset DataFrame ready for feature engineering.
    """
    log.info("=== Starting clean_dataframe ===  input shape: %s", df.shape)
    df = remove_duplicates(df)
    df = standardise_categoricals(df)
    df = validate_ranges(df)
    df = handle_missing(df)
    df = df.reset_index(drop=True)
    log.info("=== clean_dataframe complete ===  output shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 5 — sklearn ColumnTransformer  (numeric + categorical pipelines)
# ---------------------------------------------------------------------------

def build_preprocessor(
    scale_numeric: bool = True,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer for the insurance feature set.

    Parameters
    ----------
    scale_numeric : bool
        True  → apply StandardScaler after median imputation.
                 Required for Linear Regression, Ridge, Lasso, SVR.
        False → skip scaling.
                 Preferred for tree-based models (RF, GBR, DT).
    numeric_features : list of str, optional
        Numeric feature names (defaults to NUMERIC_FEATURES).
    categorical_features : list of str, optional
        Categorical feature names (defaults to CATEGORICAL_FEATURES).

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    # Numeric pipeline
    numeric_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",           # drop first level → avoids dummy variable trap
            handle_unknown="ignore",# safe for inference on new region/sex values
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",   # silently ignore any extra columns
        verbose_feature_names_out=False,
    )
    return preprocessor



def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract human-readable feature names from a *fitted* ColumnTransformer.

    Returns the numeric column names followed by the OHE category names,
    in the same order as the transformed matrix columns.
    """
    names: list[str] = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "num":
            names.extend(columns)
        elif transformer_name == "cat":
            encoder: OneHotEncoder = transformer.named_steps["encoder"]
            names.extend(encoder.get_feature_names_out(columns).tolist())

    return names


# ---------------------------------------------------------------------------
# Step 6 — Train / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    log_transform_target: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the cleaned dataframe into stratified train and test sets.

    Stratification is done on a binned version of `smoker × charges_quartile`
    so that both smoker ratio and the charge distribution are preserved in
    each split — important because smokers make up only ~20 % of the data
    and drive the highest charges.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of clean_dataframe).
    test_size : float
        Fraction reserved for testing (default 0.20 → 80/20 split).
    log_transform_target : bool
        If True, apply np.log1p to charges before splitting.
        The train pipeline uses this when the model benefits from a
        more normal target distribution (e.g. Linear Regression).

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : pd.Series
        Target vectors (raw or log-transformed depending on flag).
    """
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    if log_transform_target:
        y = np.log1p(y)
        log.info("split_data: applied log1p transform to target.")

    # Build a stratification key: smoker status + charge quartile
    # This ensures ~20 % smokers and balanced charge distribution in each split
    charge_quartile = pd.qcut(df[TARGET_COLUMN], q=4, labels=False)
    strat_key = df["smoker"].astype(str) + "_q" + charge_quartile.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=strat_key,
    )

    log.info(
        "split_data: train=%d rows  test=%d rows  (%.0f / %.0f split)",
        len(X_train), len(X_test),
        (1 - test_size) * 100, test_size * 100,
    )

    # Verify smoker ratio is preserved
    if "smoker" in X_train.columns:
        train_smoker_pct = (X_train["smoker"] == "yes").mean() * 100
        test_smoker_pct  = (X_test["smoker"]  == "yes").mean() * 100
        log.info(
            "split_data: smoker %% — train=%.1f%%  test=%.1f%%",
            train_smoker_pct, test_smoker_pct,
        )

    return X_train, X_test, y_train, y_test
