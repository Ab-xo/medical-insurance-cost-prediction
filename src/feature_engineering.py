"""
Feature engineering for Medical Insurance Cost prediction.

Derived features are created using ONLY input features available at prediction
time — no target leakage. All transformations must be deterministic and
reproducible in production inference.

Feature strategy (based on domain knowledge + data_understanding.md):
  1. age_group         — categorical age bands (young/middle/senior)
  2. bmi_category      — WHO classification (underweight/normal/overweight/obese)
  3. is_obese          — binary flag for BMI ≥ 30 (clinical threshold)
  4. smoker_bmi_risk   — interaction term: smoker × is_obese (highest-risk group)
  5. age_bmi_product   — nonlinear interaction: older + obese = higher costs
  6. has_children      — binary: children > 0
  7. family_size       — 1 (individual) or 2+ (family policy)

Why these features work:
  - Smokers pay 3.8× more on average (strongest predictor)
  - BMI ≥ 30 with smoking compounds risk exponentially
  - Age is nonlinear: costs rise sharply after 50
  - Children affect policy type (individual vs family)
  - Region matters due to cost-of-living and state regulations

All features are added in a single engineer_features() function so the
pipeline stays simple and auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import BMI_OBESE_THRESHOLD, get_logger

log = get_logger("feature_engineering")

# ---------------------------------------------------------------------------
# Domain-driven feature thresholds (actuarial + clinical guidelines)
# ---------------------------------------------------------------------------

AGE_BANDS = [0, 30, 50, 120]
AGE_LABELS = ["young", "middle_age", "senior"]

BMI_BANDS = [0, 18.5, 25, 30, 100]
BMI_LABELS = ["underweight", "normal", "overweight", "obese"]


# ---------------------------------------------------------------------------
# Individual feature constructors (each takes df, returns df copy)
# ---------------------------------------------------------------------------

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin age into three clinically meaningful groups.

        young       → 18–29   (low baseline risk)
        middle_age  → 30–49   (rising risk)
        senior      → 50+     (highest risk, age-related conditions)

    Produces a string column so the downstream OHE pipeline encodes it.
    """
    out = df.copy()
    out["age_group"] = pd.cut(
        out["age"],
        bins=AGE_BANDS,
        labels=AGE_LABELS,
        right=False,
    ).astype(str)
    return out


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify BMI using WHO clinical thresholds.

        underweight → BMI < 18.5
        normal      → 18.5 ≤ BMI < 25
        overweight  → 25 ≤ BMI < 30
        obese       → BMI ≥ 30

    String column → encoded by the categorical pipeline.
    """
    out = df.copy()
    out["bmi_category"] = pd.cut(
        out["bmi"],
        bins=BMI_BANDS,
        labels=BMI_LABELS,
        right=False,
    ).astype(str)
    return out


def add_is_obese(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: 1 if BMI ≥ 30, else 0.

    Kept as a separate numeric feature alongside bmi_category because
    tree-based models can exploit a clean binary split directly.
    """
    out = df.copy()
    out["is_obese"] = (out["bmi"] >= BMI_OBESE_THRESHOLD).astype(int)
    return out


def add_smoker_obese_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-risk interaction: smoker AND obese.

    Smokers with BMI ≥ 30 face compounded health risk and statistically
    incur the highest insurance charges in this dataset.

    Requires: 'smoker' column to be standardised to lowercase ('yes'/'no').
    Requires: 'is_obese' to already be computed.
    """
    out = df.copy()
    if "is_obese" not in out.columns:
        out = add_is_obese(out)
    is_smoker = (out["smoker"].astype(str).str.lower() == "yes").astype(int)
    out["smoker_obese"] = (is_smoker & out["is_obese"]).astype(int)
    return out


def add_age_bmi_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative interaction: age × BMI.

    Captures the compounding effect of ageing on metabolic conditions.
    An older individual with high BMI disproportionately drives up claims.
    Normalised by 1000 to keep the scale comparable to age and BMI.
    """
    out = df.copy()
    out["age_bmi"] = (out["age"] * out["bmi"]) / 1000.0
    return out


def add_has_children(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: 1 if the policy covers at least one child, else 0.

    More expressive for linear models than the raw count because the
    marginal cost of child 1 vs 0 differs from child 3 vs 2.
    """
    out = df.copy()
    out["has_children"] = (out["children"] > 0).astype(int)
    return out


def add_family_size_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorise policy holder count into individual vs family.

        individual → 0 dependants
        small_family → 1–2 dependants
        large_family → 3+ dependants

    String column → encoded by the categorical pipeline.
    """
    out = df.copy()
    out["family_size"] = pd.cut(
        out["children"],
        bins=[-1, 0, 2, 10],
        labels=["individual", "small_family", "large_family"],
    ).astype(str)
    return out


def add_age_smoker_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric interaction: age × smoker flag.

    Older smokers face dramatically higher charges. This interaction term
    lets linear models pick up the joint effect without needing the product
    implicitly computed by a tree split.
    """
    out = df.copy()
    is_smoker = (out["smoker"].astype(str).str.lower() == "yes").astype(int)
    out["age_smoker"] = out["age"] * is_smoker
    return out


# ---------------------------------------------------------------------------
# Master function — apply all engineering in one call
# ---------------------------------------------------------------------------

# Which new categorical columns to route through the OHE pipeline
ENGINEERED_CATEGORICAL = ["age_group", "bmi_category", "family_size"]

# Which new numeric columns to route through the scaler pipeline
ENGINEERED_NUMERIC = [
    "is_obese",
    "smoker_obese",
    "age_bmi",
    "has_children",
    "age_smoker",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps and return the enriched DataFrame.

    Designed to be called on BOTH training and inference data with identical
    results — no fitting or data-dependent statistics are used here.

    New columns added
    -----------------
    Numeric (9 total — 3 original + 6 engineered):
        age, bmi, children               ← original
        is_obese, smoker_obese,
        age_bmi, has_children,
        age_smoker                       ← engineered

    Categorical (6 total — 3 original + 3 engineered):
        sex, smoker, region              ← original
        age_group, bmi_category,
        family_size                      ← engineered

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of preprocessing.clean_dataframe).

    Returns
    -------
    pd.DataFrame
        Enriched dataframe with all original + engineered columns.
    """
    log.info("engineer_features: input shape %s", df.shape)

    out = df.copy()
    out = add_age_group(out)
    out = add_bmi_category(out)
    out = add_is_obese(out)
    out = add_smoker_obese_interaction(out)
    out = add_age_bmi_interaction(out)
    out = add_has_children(out)
    out = add_family_size_category(out)
    out = add_age_smoker_interaction(out)

    new_cols = [c for c in out.columns if c not in df.columns]
    log.info("engineer_features: added %d new features: %s", len(new_cols), new_cols)
    log.info("engineer_features: output shape %s", out.shape)

    return out


def get_all_feature_groups(df_engineered: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return (numeric_features, categorical_features) after engineering.

    Excludes the target column. Intended to be called after engineer_features()
    so the preprocessor knows which columns get scaled vs OHE'd.
    """
    from src.utils import TARGET_COLUMN

    all_cols    = [c for c in df_engineered.columns if c != TARGET_COLUMN]
    numeric     = df_engineered[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in all_cols if c not in numeric]
    return numeric, categorical
