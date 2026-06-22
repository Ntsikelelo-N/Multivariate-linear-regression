import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import EXPERIENCE_VALUE_COL, MODEL_FEATURES, TARGET_COL
from src.salary_prediction.features import (
    _add_experience_value,
    _check_required_features,
    compute_vif,
    engineer_features,
)


@pytest.fixture()
def encoded_df() -> pd.DataFrame:
    """Minimal encoded DataFrame that satisfies engineer_features requirements."""
    np.random.seed(42)
    n = 100
    years_worked = np.random.randint(0, 30, n)
    years_rank = np.random.randint(0, years_worked + 1, n)
    market = np.random.uniform(0.8, 1.3, n)

    df = pd.DataFrame(
        {
            "Annual salary": np.random.uniform(30_000, 90_000, n),
            "Years worked in current field": years_worked,
            "Years worked at current rank": years_rank,
            "Market value": market,
            "Position": np.random.randint(0, 3, n),
            "Field_Engineering": np.random.randint(0, 2, n),
            "Field_Finance": np.random.randint(0, 2, n),
            "Field_Marketing": np.random.randint(0, 2, n),
            "Gender_Male": np.random.randint(0, 2, n),
        }
    )
    return df


class TestComputeVIF:
    def test_output_has_expected_columns(self, encoded_df):
        X = encoded_df[["Field_Engineering", "Field_Finance", "Field_Marketing"]]
        result = compute_vif(X)
        assert "Feature" in result.columns
        assert "VIF" in result.columns

    def test_row_count_matches_feature_count(self, encoded_df):
        X = encoded_df[["Field_Engineering", "Field_Finance", "Field_Marketing"]]
        result = compute_vif(X)
        assert len(result) == X.shape[1]

    def test_vif_values_are_positive(self, encoded_df):
        X = encoded_df[["Field_Engineering", "Field_Finance", "Field_Marketing"]]
        result = compute_vif(X)
        assert (result["VIF"] > 0).all()

    def test_sorted_descending(self, encoded_df):
        X = encoded_df[["Field_Engineering", "Field_Finance", "Field_Marketing"]]
        result = compute_vif(X)
        assert result["VIF"].is_monotonic_decreasing


class TestAddExperienceValue:
    def test_column_created(self, encoded_df):
        result = _add_experience_value(encoded_df)
        assert EXPERIENCE_VALUE_COL in result.columns

    def test_formula_is_correct(self, encoded_df):
        """Experience value must equal (years_worked + years_rank) × market."""
        result = _add_experience_value(encoded_df)
        expected = (
            encoded_df["Years worked in current field"]
            + encoded_df["Years worked at current rank"]
        ) * encoded_df["Market value"]
        pd.testing.assert_series_equal(
            result[EXPERIENCE_VALUE_COL],
            expected,
            check_names=False,
        )

    def test_no_mutation_of_input(self, encoded_df):
        before = encoded_df.columns.tolist()
        _ = _add_experience_value(encoded_df)
        assert encoded_df.columns.tolist() == before

    def test_idempotent(self, encoded_df):
        """Calling twice should not add a duplicate column or change values."""
        first_pass = _add_experience_value(encoded_df)
        second_pass = _add_experience_value(first_pass)
        assert second_pass.columns.tolist().count(EXPERIENCE_VALUE_COL) == 1
        pd.testing.assert_series_equal(
            first_pass[EXPERIENCE_VALUE_COL],
            second_pass[EXPERIENCE_VALUE_COL],
        )


class TestEngineerFeatures:
    def test_returns_tuple_of_X_y(self, encoded_df):
        X, y = engineer_features(encoded_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_X_contains_exactly_model_features(self, encoded_df):
        X, _ = engineer_features(encoded_df)
        assert sorted(X.columns.tolist()) == sorted(MODEL_FEATURES)

    def test_y_is_annual_salary(self, encoded_df):
        _, y = engineer_features(encoded_df)
        assert y.name == TARGET_COL

    def test_row_counts_match(self, encoded_df):
        X, y = engineer_features(encoded_df)
        assert len(X) == len(y) == len(encoded_df)

    def test_no_nulls_in_output(self, encoded_df):
        X, y = engineer_features(encoded_df)
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0


class TestCheckRequiredFeatures:
    def test_raises_when_feature_missing(self, encoded_df):
        df_missing = encoded_df.drop(columns=["Field_Engineering"])
        df_missing = _add_experience_value(df_missing)
        with pytest.raises(KeyError, match="Field_Engineering"):
            _check_required_features(df_missing)

    def test_passes_when_all_features_present(self, encoded_df):
        df_with_ev = _add_experience_value(encoded_df)
        _check_required_features(df_with_ev)
