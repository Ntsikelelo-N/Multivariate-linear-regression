"""
data_loader.py — load and validate the raw salary CSV.

CHANGED: The notebook used a bare ``pd.read_csv`` with a hard-coded relative
path. Moving this into a function makes the path configurable, adds schema
validation so corrupt or truncated files fail loudly, and produces a logged
audit trail of the load operation.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum expected columns in the raw CSV.
_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"salary", "exprior", "yearsworked", "yearsrank",
     "market", "degree", "otherqual", "position", "male", "Field", "yearsabs"}
)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw salary CSV and perform schema validation.

    Parameters
    ----------
    path:
        Absolute or relative path to ``salary.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with original column names intact; no transformations
        are applied here so callers can inspect the unaltered source.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on disk.
    ValueError
        If the file is empty or missing one of the required columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns from '%s'", *df.shape, path.name)

    if df.empty:
        raise ValueError(f"Dataset at '{path}' loaded with zero rows.")

    missing_cols = _REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV is missing expected columns: {sorted(missing_cols)}"
        )

    _log_null_summary(df)
    return df


def _log_null_summary(df: pd.DataFrame) -> None:
    """Log columns that contain at least one null value."""
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if null_cols.empty:
        logger.info("No null values detected.")
    else:
        for col, count in null_cols.items():
            logger.warning("Column '%s' has %d null value(s).", col, count)
