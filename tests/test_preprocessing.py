import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.salary_prediction.preprocessing import (
    decode_categoricals,
    drop_unreliable_columns,
    encode_for_modelling,
    impute_missing_salary,
    validate_data_integrity,
)


@pytest.fixture()
def raw_row() -> dict:
    """Minimal single-row dictionary matching the raw CSV schema."""
    return {
        "salary": 50_000.0,
        "exprior": 2,
        "yearsworked": 5,
        "yearsrank": 3,
        "market": 1.10,
        "degree": 1,
        "otherqual": 0,
        "position": 2,
        "male": 1,
        "Field": 1,
        "yearsabs": 10,
    }


@pytest.fixture()
def raw_df(raw_row) -> pd.DataFrame:
    """Three-row DataFrame using the raw CSV schema."""
    rows = [
        raw_row,
        {**raw_row, "salary": 65_000.0, "position": 3, "Field": 2},
        {
            **raw_row,
            "salary": 35_000.0,
            "position": 1,
            "yearsworked": 1,
            "yearsrank": 1,
            "yearsabs": 5,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def decoded_df(raw_df) -> pd.DataFrame:
    return decode_categoricals(raw_df)


class TestDecodeCategoricals:
    def test_columns_renamed(self, raw_df):
        result = decode_categoricals(raw_df)
        assert "Annual salary" in result.columns
        assert "salary" not in result.columns

    def test_position_is_ordered_categorical(self, raw_df):
        result = decode_categoricals(raw_df)
        assert hasattr(result["Position"].dtype, "ordered")
        assert result["Position"].dtype.ordered is True

    def test_position_values_mapped(self, raw_df):
        result = decode_categoricals(raw_df)
        assert set(result["Position"].dropna()) <= {
            "Junior employee",
            "Manager",
            "Executive",
        }

    def test_field_values_mapped(self, raw_df):
        result = decode_categoricals(raw_df)
        assert set(result["Field"].dropna()) <= {
            "Engineering",
            "Finance",
            "Human resources",
            "Marketing",
        }

    def test_gender_values_mapped(self, raw_df):
        result = decode_categoricals(raw_df)
        assert set(result["Gender"].dropna()) <= {"Female", "Male"}


class TestValidateDataIntegrity:
    def test_passes_clean_data(self, decoded_df):
        """No exception should be raised on valid data."""
        validate_data_integrity(decoded_df)

    def test_raises_on_negative_salary(self, decoded_df):
        bad = decoded_df.copy()
        bad.loc[0, "Annual salary"] = -1_000.0
        with pytest.raises(ValueError, match="negative numeric values"):
            validate_data_integrity(bad)

    def test_raises_when_yearsrank_exceeds_yearsworked(self, decoded_df):
        bad = decoded_df.copy()
        # Make yearsrank > yearsworked for the first row
        bad.loc[0, "Years worked at current rank"] = (
            bad.loc[0, "Years worked in current field"] + 5
        )
        with pytest.raises(AssertionError, match="logically impossible"):
            validate_data_integrity(bad)


class TestDropUnreliableColumns:
    def test_years_absent_dropped_by_default(self, decoded_df):
        result = drop_unreliable_columns(decoded_df)
        assert "Years absent from work" not in result.columns

    def test_custom_columns_dropped(self, decoded_df):
        result = drop_unreliable_columns(decoded_df, columns=["Market value"])
        assert "Market value" not in result.columns

    def test_non_existent_column_is_a_no_op(self, decoded_df):
        original_cols = set(decoded_df.columns)
        result = drop_unreliable_columns(decoded_df, columns=["does_not_exist"])
        assert set(result.columns) == original_cols


class TestImputeMissingSalary:
    def test_no_nulls_after_imputation(self):
        """A row with a null salary must be filled.

        Three rows: two non-null Executives in Marketing (provide the reference
        mean) and one null Executive in Marketing to be imputed.
        """
        df = pd.DataFrame(
            {
                "Annual salary": [80_000.0, 70_000.0, np.nan],
                "Position": pd.Categorical(
                    ["Executive", "Executive", "Executive"],
                    categories=["Junior employee", "Manager", "Executive"],
                    ordered=True,
                ),
                "Field": pd.Categorical(
                    ["Marketing", "Marketing", "Marketing"],
                    categories=[
                        "Human resources",
                        "Engineering",
                        "Finance",
                        "Marketing",
                    ],
                ),
                "Market value": [1.0, 1.0, 0.93],
            }
        )
        result = impute_missing_salary(df)
        assert result["Annual salary"].isnull().sum() == 0

    def test_imputed_value_is_market_adjusted(self):
        """Imputed salary should equal market_value × group mean."""
        market_rate = 0.93
        group_mean = 80_000.0  # only 1 Executive-Marketing row in the group
        df = pd.DataFrame(
            {
                "Annual salary": [group_mean, np.nan],
                "Position": pd.Categorical(
                    ["Executive", "Executive"],
                    categories=["Junior employee", "Manager", "Executive"],
                    ordered=True,
                ),
                "Field": pd.Categorical(
                    ["Marketing", "Marketing"],
                    categories=[
                        "Human resources",
                        "Engineering",
                        "Finance",
                        "Marketing",
                    ],
                ),
                "Market value": [1.0, market_rate],
            }
        )
        result = impute_missing_salary(df)
        expected = market_rate * group_mean
        assert abs(result.loc[1, "Annual salary"] - expected) < 1.0

    def test_no_op_when_no_nulls(self):
        """DataFrame with no nulls should be returned unchanged."""
        df = pd.DataFrame(
            {
                "Annual salary": [50_000.0, 60_000.0],
                "Position": pd.Categorical(
                    ["Manager", "Executive"],
                    categories=["Junior employee", "Manager", "Executive"],
                    ordered=True,
                ),
                "Field": pd.Categorical(
                    ["Engineering", "Marketing"],
                    categories=[
                        "Human resources",
                        "Engineering",
                        "Finance",
                        "Marketing",
                    ],
                ),
                "Market value": [1.0, 0.93],
            }
        )
        result = impute_missing_salary(df)
        pd.testing.assert_series_equal(result["Annual salary"], df["Annual salary"])


class TestEncodeForModelling:
    @pytest.fixture()
    def clean_df(self, decoded_df):
        return drop_unreliable_columns(decoded_df)

    def test_field_dummies_created(self, clean_df):
        result = encode_for_modelling(clean_df)
        dummy_cols = [c for c in result.columns if c.startswith("Field_")]
        assert len(dummy_cols) >= 1, "Expected at least one Field_ dummy column"

    def test_no_boolean_columns_remain(self, clean_df):
        """statsmodels raises on bool dtype; all dummies must be cast to int."""
        result = encode_for_modelling(clean_df)
        bool_cols = result.select_dtypes(include="bool").columns.tolist()
        assert bool_cols == [], f"Boolean columns still present: {bool_cols}"

    def test_position_is_numeric(self, clean_df):
        result = encode_for_modelling(clean_df)
        assert pd.api.types.is_numeric_dtype(result["Position"])

    def test_original_df_not_mutated(self, clean_df):
        """Preprocessing must not modify the input DataFrame."""
        col_before = clean_df.columns.tolist()
        _ = encode_for_modelling(clean_df)
        assert clean_df.columns.tolist() == col_before
