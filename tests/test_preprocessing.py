"""
Unit tests for src/preprocessing.py and src/feature_engineering.py.

Run from the project root:
    pytest tests/ -v
or:
    $env:PYTHONPATH = "."; pytest tests/ -v   # Windows PowerShell
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    clean_dataframe,
    handle_missing,
    remove_duplicates,
    split_data,
    standardise_categoricals,
    validate_ranges,
)
from src.feature_engineering import engineer_features

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """
    A small, well-formed DataFrame that mirrors the insurance.csv schema.
    5 unique rows + 1 exact duplicate.
    """
    data = {
        "age":      [25, 35, 45, 55, 60,  25],   # last row is duplicate of first
        "sex":      ["male", "female", "male", "female", "male", "male"],
        "bmi":      [22.0, 28.5, 32.0, 27.0, 35.5, 22.0],
        "children": [0, 2, 1, 3, 0, 0],
        "smoker":   ["no", "no", "yes", "no", "yes", "no"],
        "region":   ["northeast", "northwest", "southeast", "southwest", "northeast", "northeast"],
        "charges":  [3000.0, 8500.0, 25000.0, 7200.0, 40000.0, 3000.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def clean_small(minimal_df) -> pd.DataFrame:
    """Cleaned version of minimal_df (no duplicates, strings normalised)."""
    return clean_dataframe(minimal_df)


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """DataFrame with mixed-case strings, whitespace, and out-of-range values."""
    data = {
        "age":      [25,   -5,   200,  35],   # -5 and 200 are out of range
        "sex":      ["Male ", "FEMALE", " male", "Female"],
        "bmi":      [22.0,  28.5,   5.0,  95.0],  # 5.0 and 95.0 are out of range
        "children": [0,     2,      1,    3],
        "smoker":   ["No",  "YES",  "no", "Yes"],
        "region":   ["Northeast", "NORTHWEST", "southeast", "SouthWest"],
        "charges":  [3000.0, 8500.0, 15000.0, 7200.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def missing_df() -> pd.DataFrame:
    """DataFrame with NaN values in numeric and categorical columns."""
    data = {
        "age":      [25.0,  np.nan, 45.0],
        "sex":      ["male", "female", None],
        "bmi":      [22.0,   28.5,   np.nan],
        "children": [0.0,    2.0,    1.0],
        "smoker":   ["no",   "yes",   "no"],
        "region":   ["northeast", "northwest", None],
        "charges":  [3000.0, 8500.0, 15000.0],
    }
    return pd.DataFrame(data)


# ===========================================================================
# remove_duplicates
# ===========================================================================

class TestRemoveDuplicates:
    def test_removes_one_duplicate(self, minimal_df):
        result = remove_duplicates(minimal_df)
        assert len(result) == 5, "Should have 5 rows after dropping 1 duplicate"

    def test_no_duplicates_returns_same_length(self, minimal_df):
        unique_df = minimal_df.drop_duplicates()
        result = remove_duplicates(unique_df)
        assert len(result) == len(unique_df)

    def test_index_is_reset(self, minimal_df):
        result = remove_duplicates(minimal_df)
        assert list(result.index) == list(range(len(result)))

    def test_empty_dataframe(self):
        result = remove_duplicates(pd.DataFrame(columns=["a", "b"]))
        assert len(result) == 0


# ===========================================================================
# standardise_categoricals
# ===========================================================================

class TestStandardiseCategoricals:
    def test_lowercases_strings(self, dirty_df):
        result = standardise_categoricals(dirty_df)
        for col in ["sex", "smoker", "region"]:
            assert result[col].str.islower().all() or result[col].isna().all(), \
                f"Column '{col}' should be all lowercase"

    def test_strips_whitespace(self, dirty_df):
        result = standardise_categoricals(dirty_df)
        for col in ["sex", "smoker", "region"]:
            vals = result[col].dropna()
            assert (vals == vals.str.strip()).all(), \
                f"Column '{col}' should have no leading/trailing whitespace"

    def test_does_not_modify_numerics(self, dirty_df):
        original_age = dirty_df["age"].copy()
        result = standardise_categoricals(dirty_df)
        pd.testing.assert_series_equal(result["age"], original_age)

    def test_known_values_correct(self, dirty_df):
        result = standardise_categoricals(dirty_df)
        assert result["sex"].iloc[0]    == "male"
        assert result["smoker"].iloc[1] == "yes"
        assert result["region"].iloc[0] == "northeast"


# ===========================================================================
# validate_ranges
# ===========================================================================

class TestValidateRanges:
    def test_negative_age_becomes_nan(self, dirty_df):
        result = validate_ranges(standardise_categoricals(dirty_df))
        assert np.isnan(result["age"].iloc[1]), "age=-5 should be NaN"

    def test_excessive_age_becomes_nan(self, dirty_df):
        result = validate_ranges(standardise_categoricals(dirty_df))
        assert np.isnan(result["age"].iloc[2]), "age=200 should be NaN"

    def test_low_bmi_becomes_nan(self, dirty_df):
        result = validate_ranges(standardise_categoricals(dirty_df))
        assert np.isnan(result["bmi"].iloc[2]), "bmi=5 should be NaN"

    def test_high_bmi_becomes_nan(self, dirty_df):
        result = validate_ranges(standardise_categoricals(dirty_df))
        assert np.isnan(result["bmi"].iloc[3]), "bmi=95 should be NaN"

    def test_valid_values_unchanged(self, dirty_df):
        result = validate_ranges(dirty_df)
        assert result["age"].iloc[0] == 25.0
        assert result["bmi"].iloc[0] == 22.0


# ===========================================================================
# handle_missing
# ===========================================================================

class TestHandleMissing:
    def test_numeric_nans_are_filled(self, missing_df):
        result = handle_missing(missing_df)
        assert result["age"].isna().sum() == 0
        assert result["bmi"].isna().sum() == 0

    def test_categorical_nans_are_filled(self, missing_df):
        result = handle_missing(missing_df)
        assert result["sex"].isna().sum() == 0
        assert result["region"].isna().sum() == 0

    def test_no_missing_target(self):
        df_with_null_target = pd.DataFrame(
            {
                "age": [25, 30],
                "sex": ["male", "female"],
                "bmi": [22.0, 28.0],
                "children": [0, 1],
                "smoker": ["no", "yes"],
                "region": ["northeast", "northwest"],
                "charges": [None, 8000.0],
            }
        )
        result = handle_missing(df_with_null_target)
        # Row with null target should be dropped
        assert result["charges"].isna().sum() == 0
        assert len(result) == 1


# ===========================================================================
# clean_dataframe  (integration)
# ===========================================================================

class TestCleanDataframe:
    def test_output_shape_after_dedup(self, minimal_df):
        """clean_dataframe removes 1 duplicate → 5 rows remain."""
        result = clean_dataframe(minimal_df)
        assert len(result) == 5

    def test_output_has_correct_columns(self, minimal_df):
        result = clean_dataframe(minimal_df)
        expected_cols = set(minimal_df.columns)
        assert set(result.columns) == expected_cols

    def test_no_missing_values_in_clean_output(self, minimal_df):
        result = clean_dataframe(minimal_df)
        assert result.isna().sum().sum() == 0

    def test_strings_are_lowercase(self, dirty_df):
        result = clean_dataframe(dirty_df)
        for col in ["sex", "smoker", "region"]:
            assert result[col].str.islower().all()

    def test_index_is_reset(self, minimal_df):
        result = clean_dataframe(minimal_df)
        assert list(result.index) == list(range(len(result)))


# ===========================================================================
# split_data
# ===========================================================================

@pytest.fixture
def split_ready_df() -> pd.DataFrame:
    """
    A 50-row synthetic DataFrame large enough for the stratified split
    (each smoker×quartile stratum needs ≥ 2 members).
    """
    rng = np.random.default_rng(0)
    n = 50
    smoker_arr = rng.choice(["yes", "no"], n, p=[0.2, 0.8])
    # Make charges correlated with smoker so quartiles create balanced strata
    charges = np.where(
        smoker_arr == "yes",
        rng.uniform(20000, 60000, n),
        rng.uniform(1000, 15000, n),
    ).round(2)
    return pd.DataFrame(
        {
            "age":      rng.integers(18, 65, n).tolist(),
            "sex":      rng.choice(["male", "female"], n).tolist(),
            "bmi":      rng.uniform(18, 45, n).round(1).tolist(),
            "children": rng.integers(0, 5, n).tolist(),
            "smoker":   smoker_arr.tolist(),
            "region":   rng.choice(["northeast", "northwest", "southeast", "southwest"], n).tolist(),
            "charges":  charges.tolist(),
        }
    )


class TestSplitData:
    def test_output_sizes_sum_to_input(self, split_ready_df):
        X_train, X_test, y_train, y_test = split_data(split_ready_df)
        total = len(X_train) + len(X_test)
        assert total == len(split_ready_df)

    def test_default_test_size_is_20pct(self, split_ready_df):
        X_train, X_test, y_train, y_test = split_data(split_ready_df)
        ratio = len(X_test) / len(split_ready_df)
        # Allow ± 0.10 tolerance due to integer rounding on small datasets
        assert abs(ratio - 0.20) <= 0.10

    def test_target_not_in_feature_columns(self, split_ready_df):
        X_train, X_test, y_train, y_test = split_data(split_ready_df)
        assert "charges" not in X_train.columns
        assert "charges" not in X_test.columns

    def test_y_series_name(self, split_ready_df):
        _, _, y_train, y_test = split_data(split_ready_df)
        assert y_train.name == "charges"
        assert y_test.name  == "charges"

    def test_larger_dataset_split_ratio(self):
        """Test split ratio on a larger synthetic dataset."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "age":      rng.integers(18, 65, n),
                "sex":      rng.choice(["male", "female"], n),
                "bmi":      rng.uniform(18, 45, n).round(1),
                "children": rng.integers(0, 5, n),
                "smoker":   rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
                "region":   rng.choice(["northeast", "northwest", "southeast", "southwest"], n),
                "charges":  rng.uniform(1000, 60000, n).round(2),
            }
        )
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.20)
        ratio = len(X_test) / n
        assert abs(ratio - 0.20) <= 0.05

    def test_smoker_ratio_preserved(self):
        """Stratified split should keep smoker ratio within ±5% across splits."""
        rng = np.random.default_rng(42)
        n = 500
        smoker_arr = rng.choice(["yes", "no"], n, p=[0.2, 0.8])
        df = pd.DataFrame(
            {
                "age":      rng.integers(18, 65, n),
                "sex":      rng.choice(["male", "female"], n),
                "bmi":      rng.uniform(18, 45, n).round(1),
                "children": rng.integers(0, 5, n),
                "smoker":   smoker_arr,
                "region":   rng.choice(["northeast", "northwest", "southeast", "southwest"], n),
                "charges":  rng.uniform(1000, 60000, n).round(2),
            }
        )
        X_train, X_test, _, _ = split_data(df, test_size=0.20)
        original_pct = (df["smoker"] == "yes").mean()
        train_pct    = (X_train["smoker"] == "yes").mean()
        test_pct     = (X_test["smoker"]  == "yes").mean()
        assert abs(train_pct - original_pct) <= 0.05
        assert abs(test_pct  - original_pct) <= 0.05


# ===========================================================================
# engineer_features
# ===========================================================================

class TestEngineerFeatures:
    EXPECTED_NEW_COLS = [
        "age_group",
        "bmi_category",
        "is_obese",
        "smoker_obese",
        "age_bmi",
        "has_children",
        "family_size",
        "age_smoker",
    ]

    def test_adds_all_expected_columns(self, clean_small):
        result = engineer_features(clean_small)
        for col in self.EXPECTED_NEW_COLS:
            assert col in result.columns, f"Missing engineered column: '{col}'"

    def test_output_has_more_columns_than_input(self, clean_small):
        result = engineer_features(clean_small)
        assert len(result.columns) > len(clean_small.columns)

    def test_preserves_original_rows(self, clean_small):
        result = engineer_features(clean_small)
        assert len(result) == len(clean_small)

    def test_is_obese_correct_threshold(self, clean_small):
        result = engineer_features(clean_small)
        for _, row in result.iterrows():
            expected = 1 if row["bmi"] >= 30 else 0
            assert row["is_obese"] == expected, \
                f"is_obese wrong for BMI={row['bmi']}"

    def test_smoker_obese_interaction(self, clean_small):
        result = engineer_features(clean_small)
        for _, row in result.iterrows():
            expected = 1 if (row["smoker"] == "yes" and row["is_obese"] == 1) else 0
            assert row["smoker_obese"] == expected

    def test_has_children_binary(self, clean_small):
        result = engineer_features(clean_small)
        assert set(result["has_children"].unique()).issubset({0, 1})

    def test_age_group_values(self, clean_small):
        result = engineer_features(clean_small)
        valid = {"young", "middle_age", "senior"}
        actual = set(result["age_group"].unique())
        assert actual.issubset(valid), f"Unexpected age_group values: {actual - valid}"

    def test_bmi_category_values(self, clean_small):
        result = engineer_features(clean_small)
        valid = {"underweight", "normal", "overweight", "obese"}
        actual = set(result["bmi_category"].unique())
        assert actual.issubset(valid), f"Unexpected bmi_category values: {actual - valid}"

    def test_age_bmi_product(self, clean_small):
        result = engineer_features(clean_small)
        expected = (clean_small["age"] * clean_small["bmi"] / 1000.0).values
        np.testing.assert_allclose(result["age_bmi"].values, expected, rtol=1e-5)

    def test_no_target_leakage(self, clean_small):
        """engineer_features should not use the target column in any computation."""
        # Modify charges — engineered features should be identical
        df_modified = clean_small.copy()
        df_modified["charges"] = df_modified["charges"] * 999

        r1 = engineer_features(clean_small)
        r2 = engineer_features(df_modified)

        for col in self.EXPECTED_NEW_COLS:
            if r1[col].dtype in [np.float64, np.int64, float, int]:
                np.testing.assert_array_equal(
                    r1[col].values, r2[col].values,
                    err_msg=f"Column '{col}' differs when charges are changed — possible leakage"
                )
            else:
                assert list(r1[col]) == list(r2[col]), \
                    f"Column '{col}' differs when charges are changed — possible leakage"

    def test_idempotent(self, clean_small):
        """Calling engineer_features twice should not add duplicate columns."""
        once  = engineer_features(clean_small)
        twice = engineer_features(once)
        # The second call overwrites; column count must equal the first
        assert set(once.columns) == set(twice.columns)
