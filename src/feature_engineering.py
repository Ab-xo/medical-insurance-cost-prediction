"""Feature engineering — adds 8 domain-driven columns on top of the 6 raw inputs.

All features use only input columns — no target leakage.
Safe to call identically on both training and inference data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import BMI_OBESE_THRESHOLD, get_logger

log = get_logger("feature_engineering")

# Age and BMI bin definitions
AGE_BANDS  = [0, 30, 50, 120]
AGE_LABELS = ["young", "middle_age", "senior"]

BMI_BANDS  = [0, 18.5, 25, 30, 100]
BMI_LABELS = ["underweight", "normal", "overweight", "obese"]

# New column names exposed for downstream pipeline routing
ENGINEERED_CATEGORICAL = ["age_group", "bmi_category", "family_size"]
ENGINEERED_NUMERIC     = ["is_obese", "smoker_obese", "age_bmi", "has_children", "age_smoker"]


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin age into young / middle_age / senior."""
    out = df.copy()
    out["age_group"] = pd.cut(out["age"], bins=AGE_BANDS, labels=AGE_LABELS, right=False).astype(str)
    return out


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Classify BMI using WHO thresholds: underweight / normal / overweight / obese."""
    out = df.copy()
    out["bmi_category"] = pd.cut(out["bmi"], bins=BMI_BANDS, labels=BMI_LABELS, right=False).astype(str)
    return out


def add_is_obese(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if BMI ≥ 30."""
    out = df.copy()
    out["is_obese"] = (out["bmi"] >= BMI_OBESE_THRESHOLD).astype(int)
    return out


def add_smoker_obese_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if smoker AND obese. Highest-charge group in the dataset."""
    out = df.copy()
    if "is_obese" not in out.columns:
        out = add_is_obese(out)
    is_smoker = (out["smoker"].astype(str).str.lower() == "yes").astype(int)
    out["smoker_obese"] = (is_smoker & out["is_obese"]).astype(int)
    return out


def add_age_bmi_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric interaction: age × bmi / 1000. Captures compound metabolic risk."""
    out = df.copy()
    out["age_bmi"] = (out["age"] * out["bmi"]) / 1000.0
    return out


def add_has_children(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if the policy covers at least one child."""
    out = df.copy()
    out["has_children"] = (out["children"] > 0).astype(int)
    return out


def add_family_size_category(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket dependant count: individual / small_family / large_family."""
    out = df.copy()
    out["family_size"] = pd.cut(
        out["children"],
        bins=[-1, 0, 2, 10],
        labels=["individual", "small_family", "large_family"],
    ).astype(str)
    return out


def add_age_smoker_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric interaction: age × smoker_flag. Older smokers pay substantially more."""
    out = df.copy()
    is_smoker = (out["smoker"].astype(str).str.lower() == "yes").astype(int)
    out["age_smoker"] = out["age"] * is_smoker
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 8 feature engineering steps and return the enriched DataFrame."""
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
    """Return (numeric_features, categorical_features) from an engineered DataFrame.

    Excludes the target column. Used to configure build_preprocessor().
    """
    from src.utils import TARGET_COLUMN

    all_cols    = [c for c in df_engineered.columns if c != TARGET_COLUMN]
    numeric     = df_engineered[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in all_cols if c not in numeric]
    return numeric, categorical
