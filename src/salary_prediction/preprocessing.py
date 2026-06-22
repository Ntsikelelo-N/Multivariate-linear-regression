"""
preprocessing.py — cleaning, validation, imputation, and encoding.

CHANGED: Every notebook cell that mutated a global DataFrame in-place has been
converted to a pure function that returns a *new* DataFrame (no ``inplace=True``
anywhere). This eliminates hidden side-effects, makes each step unit-testable,
and ensures the notebook can be re-run safely in any cell order.

Anti-patterns fixed
-------------------
- ``inplace=True`` throughout  → explicit reassignment / ``df.copy()``
- Global variable capture in helper functions  → explicit ``df`` argument
- Hard-coded ``0.93`` imputation scalar  → read from the actual row's market value
- Assertion in a function that swallows the context  → raises with a message
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    COLUMN_RENAME_MAP, DEGREE_MAP, OTHERQUAL_MAP, GENDER_MAP,
    POSITION_MAP, FIELD_MAP, POSITION_ORDER, FIELD_ORDER,
    IMPUTATION_POSITION, IMPUTATION_FIELD, TARGET_COL,
)

logger = logging.getLogger(__name__)


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on the raw salary DataFrame.

    Steps (in order):
    1. Decode integer-coded categoricals and rename columns.
    2. Validate business rules (non-negative values, years consistency).
    3. Drop the unreliable ``Years absent from work`` column.
    4. Impute the single missing salary using a market-adjusted group mean.
    5. Encode ordinal and nominal features for modelling.
    6. Drop low-information binary columns (degree, other qual).

    Parameters
    ----------
    raw_df:
        Output of :func:`salary_prediction.data_loader.load_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned and encoded DataFrame ready for feature engineering.
    """
    df = decode_categoricals(raw_df)
    validate_data_integrity(df)
    df = drop_unreliable_columns(df)
    df = impute_missing_salary(df)
    df = encode_for_modelling(df)
    logger.info("Preprocessing complete: %d rows × %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 1 — Decode categoricals
# ---------------------------------------------------------------------------

def decode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw integer columns to human-readable labels and set dtypes.

    The raw CSV encodes position, field, gender, and qualification as
    integers. This function maps them to their semantic labels (defined in
    config.py) and casts each to an ordered or unordered Categorical, which
    prevents accidental numeric operations on ordinal codes.
    """
    df = df.copy()
    df = df.rename(columns=COLUMN_RENAME_MAP)

    df["Has degree"] = (
        df["Has degree"].map(DEGREE_MAP).astype("category")
    )
    df["Has other post-secondary qualification"] = (
        df["Has other post-secondary qualification"].map(OTHERQUAL_MAP).astype("category")
    )
    df["Gender"] = df["Gender"].map(GENDER_MAP).astype("category")
    df["Position"] = pd.Categorical(
        df["Position"].map(POSITION_MAP),
        categories=POSITION_ORDER,
        ordered=True,
    )
    df["Field"] = df["Field"].map(FIELD_MAP).astype("category")

    logger.debug("Categoricals decoded; columns now: %s", df.columns.tolist())
    return df


# ---------------------------------------------------------------------------
# Step 2 — Validate business rules
# ---------------------------------------------------------------------------

def validate_data_integrity(df: pd.DataFrame) -> None:
    """Assert domain constraints and log warnings for anomalies.

    This function is deliberately read-only — it raises on hard violations
    and logs warnings for soft anomalies so the caller can decide whether
    to proceed.

    Raises
    ------
    ValueError
        If any numeric salary or experience value is negative.
    AssertionError
        If ``yearsworked < yearsrank`` for any row (logically impossible).
    """
    years_worked_col = "Years worked in current field"
    years_rank_col   = "Years worked at current rank"

    # Hard constraint: no negative numeric values
    numeric_cols = df.select_dtypes(include="number")
    non_negative = (numeric_cols.dropna() >= 0).all().all()
    if not non_negative:
        raise ValueError(
            "Dataset contains negative numeric values — check for data corruption."
        )

    # Hard constraint: yearsworked ≥ yearsrank
    invalid_experience = df[years_worked_col] < df[years_rank_col]
    if invalid_experience.any():
        bad_rows = int(invalid_experience.sum())
        raise AssertionError(
            f"{bad_rows} row(s) have 'years in field' < 'years at rank', "
            "which is logically impossible."
        )

    # Soft constraint: flag suspicious 'years absent' values
    years_absent_col = "Years absent from work"
    if years_absent_col in df.columns:
        prior_col = "Years of experience in prior field"
        implausible = (
            df[prior_col] + df[years_worked_col] < df[years_absent_col]
        )
        n_implausible = int(implausible.sum())
        if n_implausible > 0:
            logger.warning(
                "%d rows have 'years absent' exceeding total career length. "
                "Column will be dropped in the next step.",
                n_implausible,
            )


# ---------------------------------------------------------------------------
# Step 3 — Drop unreliable columns
# ---------------------------------------------------------------------------

def drop_unreliable_columns(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Drop columns that carry unreliable or low-information signal.

    The ``Years absent from work`` column contains values implausibly large
    (> 70) alongside zeros for employees with 30+ year tenures, suggesting the
    unit of measurement is ambiguous. It is dropped by default.

    Parameters
    ----------
    df:
        Cleaned DataFrame.
    columns:
        Override the list of columns to drop. Defaults to
        ``["Years absent from work"]``.
    """
    if columns is None:
        columns = ["Years absent from work"]

    existing = [c for c in columns if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        logger.info("Dropped columns: %s", existing)
    return df


# ---------------------------------------------------------------------------
# Step 4 — Impute missing salary
# ---------------------------------------------------------------------------

def impute_missing_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the single missing salary with a market-adjusted group mean.

    Strategy: the missing row is an Executive in Marketing. Rather than
    using a flat group mean (which ignores that employee's personal market
    rating), we multiply the group mean by the row's own market value.
    This is equivalent to: ``predicted_salary = group_mean × market_rate``.

    The original notebook hard-coded ``0.93`` — this implementation reads the
    actual market value from the row so the logic is data-driven.
    """
    missing_mask = df[TARGET_COL].isnull()
    n_missing = int(missing_mask.sum())

    if n_missing == 0:
        logger.info("No missing salary values; imputation skipped.")
        return df

    logger.info("Imputing %d missing salary value(s).", n_missing)

    group_mean = df.loc[
        (df["Position"] == IMPUTATION_POSITION) & (df["Field"] == IMPUTATION_FIELD),
        TARGET_COL,
    ].mean()

    # For each missing row, scale the group mean by the row's own market rate
    df = df.copy()
    missing_indices = df.index[missing_mask]
    for idx in missing_indices:
        market_rate = df.at[idx, "Market value"]
        imputed_value = market_rate * group_mean
        df.at[idx, TARGET_COL] = imputed_value
        logger.debug(
            "Row %d: market_rate=%.4f, group_mean=%.2f, imputed=%.2f",
            idx, market_rate, group_mean, imputed_value,
        )

    return df


# ---------------------------------------------------------------------------
# Step 5 — Encode features
# ---------------------------------------------------------------------------

def encode_for_modelling(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal fields and ordinal-encode Position.

    One-hot encoding
    ----------------
    ``Field`` is one-hot encoded with Human Resources as the reference
    category (``drop_first=True`` relative to the ordered FIELD_ORDER list).
    ``Gender`` is also encoded but excluded from the final model due to
    near-zero predictive power; it remains in the DataFrame for EDA.

    Ordinal encoding
    ----------------
    ``Position`` preserves its natural hierarchy
    (Junior employee = 0, Manager = 1, Executive = 2).

    Returns
    -------
    pd.DataFrame
        DataFrame with boolean dummy columns cast to int (0/1) so
        statsmodels does not raise on boolean dtypes.
    """
    from sklearn.preprocessing import OrdinalEncoder

    df = df.copy()

    # --- Nominal: Field and Gender ---
    for col, ordered_cats in [
        ("Field",  FIELD_ORDER),
        ("Gender", ["Female", "Male"]),
    ]:
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=ordered_cats, ordered=True)

    df = pd.get_dummies(df, columns=["Field", "Gender"], drop_first=True)

    # Cast bool dummies to int so statsmodels handles them correctly
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # --- Ordinal: Position ---
    if "Position" in df.columns:
        enc = OrdinalEncoder(categories=[POSITION_ORDER])
        df["Position"] = enc.fit_transform(df[["Position"]]).astype(int)

    # --- Drop low-information binary columns ---
    low_info = [
        "Has degree",
        "Has other post-secondary qualification",
    ]
    existing_low_info = [c for c in low_info if c in df.columns]
    if existing_low_info:
        df = df.drop(columns=existing_low_info)
        logger.info(
            "Dropped low-information columns: %s "
            "(class imbalance: degree holders = 496/514, other qual = 23/514).",
            existing_low_info,
        )

    return df
