"""
features.py — feature engineering and multicollinearity analysis.

CHANGED: The notebook scattered VIF computation, composite feature creation,
and feature selection across ~10 cells with global state. This module
collects them into three focused functions with clear inputs and outputs.

Key design decision: ``engineer_features`` is deterministic and stateless —
given the same preprocessed DataFrame it always returns the same feature
matrix, making it trivially testable.
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    EXPERIENCE_VALUE_COL,
    MODEL_FEATURES,
    TARGET_COL,
    VIF_THRESHOLD,
)

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model feature matrix and target vector.

    Constructs ``Experience value = (years_in_field + years_at_rank) × market``
    which consolidates three individually collinear predictors into a single
    composite that passes VIF < 5 while retaining most of their explanatory
    power.

    Parameters
    ----------
    df:
        Output of :func:`salary_prediction.preprocessing.preprocess`.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing ``MODEL_FEATURES`` (defined in config.py).
    y : pd.Series
        Target vector — ``Annual salary``.
    """
    df = _add_experience_value(df)
    _check_required_features(df)

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    vif_summary = compute_vif(X)
    high_vif = vif_summary[vif_summary["VIF"] > VIF_THRESHOLD]
    if not high_vif.empty:
        logger.warning(
            "Features with VIF > %.1f (may cause unstable estimates): %s",
            VIF_THRESHOLD,
            high_vif["Feature"].tolist(),
        )

    logger.info(
        "Feature matrix: %d rows × %d features. Target: '%s'.",
        *X.shape, TARGET_COL,
    )
    return X, y


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute the Variance Inflation Factor for each column in X.

    Parameters
    ----------
    X:
        Feature matrix (numeric only, no target column, no constant).

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with ``Feature`` and ``VIF`` columns, sorted
        descending by VIF so the most collinear features surface first.

    Notes
    -----
    A constant column is added internally for the VIF calculation and removed
    afterward; callers should not add a constant before calling this function.
    """
    X_with_const = X.copy()
    X_with_const.insert(0, "_const", 1.0)

    vif_values = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(1, X_with_const.shape[1])  # skip the synthetic constant
    ]

    return (
        pd.DataFrame({"Feature": X.columns, "VIF": vif_values})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def compute_feature_correlations(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    min_abs_corr: float = 0.30,
    max_p_value: float = 0.05,
) -> pd.DataFrame:
    """Return pairwise Pearson correlations and p-values for numeric columns.

    Parameters
    ----------
    df:
        Encoded DataFrame (output of preprocessing).
    target:
        Column name of the prediction target; used to flag target-vs-feature
        rows separately.
    min_abs_corr:
        Filter: only return rows with |correlation| ≥ this value.
    max_p_value:
        Filter: only return rows with p-value ≤ this value.

    Returns
    -------
    pd.DataFrame
        Columns: ``Feature 1``, ``Feature 2``, ``Pearson r``, ``p-value``.
    """
    from scipy.stats import pearsonr

    numeric_df = df.select_dtypes(include="number").dropna()
    cols = numeric_df.columns.tolist()

    records = []
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            corr, p_val = pearsonr(numeric_df[col_a], numeric_df[col_b])
            records.append(
                {"Feature 1": col_a, "Feature 2": col_b,
                 "Pearson r": corr, "p-value": p_val}
            )

    result = pd.DataFrame(records)
    mask = (
        (result["p-value"] <= max_p_value) &
        (result["Pearson r"].abs() >= min_abs_corr)
    )
    return result[mask].sort_values("Pearson r", key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_experience_value(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the experience-value composite and append it as a new column.

    Formula
    -------
    ``Experience value = (years_in_field + years_at_rank) × market_value``

    Rationale: both tenure columns are individually correlated with salary
    (r ≈ 0.62 and 0.61) and with each other (r ≈ 0.70), producing VIF > 5
    when included separately. Summing them and weighting by the employee's
    market rate produces a single composite that captures the joint signal
    while reducing collinearity.
    """
    years_col  = "Years worked in current field"
    rank_col   = "Years worked at current rank"
    market_col = "Market value"

    if EXPERIENCE_VALUE_COL in df.columns:
        logger.debug("'%s' already present; skipping computation.", EXPERIENCE_VALUE_COL)
        return df

    df = df.copy()
    df[EXPERIENCE_VALUE_COL] = (
        (df[years_col] + df[rank_col]) * df[market_col]
    )
    logger.info("Engineered feature '%s' added.", EXPERIENCE_VALUE_COL)
    return df


def _check_required_features(df: pd.DataFrame) -> None:
    """Raise KeyError if any MODEL_FEATURES column is absent."""
    missing = [col for col in MODEL_FEATURES if col not in df.columns]
    if missing:
        raise KeyError(
            f"Required model features are missing from the DataFrame: {missing}. "
            "Ensure preprocessing and feature engineering have run first."
        )
