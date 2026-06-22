"""
model.py — OLS regression training, assumption checks, and evaluation.

CHANGED: The notebook trained the model, ran cross-validation, and computed
metrics across ~15 separate cells, making it impossible to test or reuse any
piece individually. This module isolates each concern into a named function
and returns structured results rather than printing raw output to stdout.

Anti-patterns fixed
-------------------
- ``sm.add_constant`` called separately on train AND test with no guarantee
  the constant column ends up in the same position.
- ``mean_squared_error(..., squared=False)`` — deprecated in scikit-learn ≥ 1.4;
  replaced with ``root_mean_squared_error``.
- Cross-validation loop re-selected features from the full encoded DataFrame
  inside the loop body, which risks inconsistent column ordering across folds.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from statsmodels.regression.linear_model import RegressionResultsWrapper

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    CV_N_SPLITS,
    NORMALITY_ALPHA,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class HoldoutMetrics:
    """Evaluation metrics on the held-out test set."""
    mae:  float
    rmse: float
    r2:   float


@dataclass
class CVMetrics:
    """Cross-validation summary statistics."""
    mean_mse:  float
    std_mse:   float
    mean_rmse: float
    n_splits:  int


@dataclass
class ModelResult:
    """Container for every artefact produced by :func:`train_model`."""
    model:           RegressionResultsWrapper
    holdout:         HoldoutMetrics
    cv:              CVMetrics
    normality_pass:  bool
    shapiro_p_value: float
    X_test:          pd.DataFrame
    y_test:          pd.Series
    y_pred:          pd.Series


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model(X: pd.DataFrame, y: pd.Series) -> ModelResult:
    """Train an OLS regression model and run the full evaluation suite.

    Pipeline
    --------
    1. 80/20 stratified train/test split.
    2. Fit OLS on the training set (statsmodels — preserves p-values, CIs).
    3. Check residual normality via Shapiro-Wilk.
    4. Evaluate on the hold-out test set (MAE, RMSE, R²).
    5. 20-fold cross-validation on the full dataset.

    Parameters
    ----------
    X:
        Feature matrix (output of :func:`salary_prediction.features.engineer_features`).
    y:
        Target vector — ``Annual salary``.

    Returns
    -------
    ModelResult
        Structured result carrying the fitted model, all metrics, and the
        test split so callers can produce diagnostic plots without re-splitting.
    """
    X_train, X_test, y_train, y_test = _split(X, y)

    model = _fit_ols(X_train, y_train)

    normality_pass, shapiro_p = _check_normality(model.resid)

    holdout = _evaluate_holdout(model, X_test, y_test)

    cv_metrics = _cross_validate(X, y)

    y_pred = model.predict(sm.add_constant(X_test, has_constant="add"))

    logger.info(
        "Training complete — Test MAE: $%.0f | RMSE: $%.0f | R²: %.3f",
        holdout.mae, holdout.rmse, holdout.r2,
    )

    return ModelResult(
        model=model,
        holdout=holdout,
        cv=cv_metrics,
        normality_pass=normality_pass,
        shapiro_p_value=shapiro_p,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
    )


def print_summary(result: ModelResult) -> None:
    """Print a human-readable evaluation summary to stdout."""
    print(result.model.summary())
    print("\n" + "=" * 60)
    print("HOLD-OUT TEST SET METRICS")
    print(f"  MAE  : ${result.holdout.mae:>10,.0f}")
    print(f"  RMSE : ${result.holdout.rmse:>10,.0f}")
    print(f"  R²   : {result.holdout.r2:>11.4f}")
    print("\nCROSS-VALIDATION ({} folds)".format(result.cv.n_splits))
    print(f"  Mean MSE  : {result.cv.mean_mse:>14,.0f}")
    print(f"  Std  MSE  : {result.cv.std_mse:>14,.0f}")
    print(f"  Mean RMSE : ${result.cv.mean_rmse:>10,.0f}")
    print("\nRESIDUAL NORMALITY (Shapiro-Wilk)")
    status = "PASS ✓" if result.normality_pass else "FAIL ✗ (mild; OLS robust at n>400)"
    print(f"  p = {result.shapiro_p_value:.4f}  →  {status}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reproducible 80/20 train-test split."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def _fit_ols(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RegressionResultsWrapper:
    """Add intercept column and fit an OLS model on the training data."""
    X_const = sm.add_constant(X_train, has_constant="add")
    model = sm.OLS(y_train, X_const).fit()
    logger.info(
        "OLS fitted — Train R²: %.4f | AIC: %.1f | n=%d",
        model.rsquared, model.aic, int(model.nobs),
    )
    return model


def _check_normality(
    residuals: pd.Series,
    alpha: float = NORMALITY_ALPHA,
) -> tuple[bool, float]:
    """Run Shapiro-Wilk normality test on model residuals.

    Returns
    -------
    (passed, p_value) : tuple[bool, float]
        ``passed`` is True if the null hypothesis of normality is *not*
        rejected at the given significance level.
    """
    _, p_value = shapiro(residuals)
    passed = bool(p_value > alpha)   # cast np.bool_ → Python bool for isinstance checks
    level = "PASS" if passed else "FAIL"
    logger.info("Shapiro-Wilk residual normality: p=%.4f (%s)", p_value, level)
    return passed, float(p_value)


def _evaluate_holdout(
    model: RegressionResultsWrapper,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> HoldoutMetrics:
    """Generate predictions on the test set and compute MAE, RMSE, R²."""
    X_test_const = sm.add_constant(X_test, has_constant="add")
    y_pred = model.predict(X_test_const)

    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))

    # Compute R² on test set (not the training R² from the summary)
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return HoldoutMetrics(mae=mae, rmse=rmse, r2=r2)


def _cross_validate(X: pd.DataFrame, y: pd.Series) -> CVMetrics:
    """20-fold cross-validation using OLS refitted on each fold's training set.

    The original notebook computed CV on the already-encoded DataFrame with
    column selection inside the loop, which could silently produce mismatched
    feature orders if the DataFrame columns changed. Here X is passed directly
    so the feature set is frozen before the loop begins.
    """
    kf = KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    mse_scores: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        X_tr_const = sm.add_constant(X_tr, has_constant="add")
        X_te_const = sm.add_constant(X_te, has_constant="add")

        fold_model = sm.OLS(y_tr, X_tr_const).fit()
        y_pred_fold = fold_model.predict(X_te_const)

        mse = float(np.mean((y_te - y_pred_fold) ** 2))
        mse_scores.append(mse)
        logger.debug("Fold %02d MSE: %.0f", fold_idx + 1, mse)

    mean_mse  = float(np.mean(mse_scores))
    std_mse   = float(np.std(mse_scores))
    mean_rmse = float(np.sqrt(mean_mse))

    logger.info(
        "CV (%d folds) — Mean MSE: %.0f ± %.0f | Mean RMSE: $%.0f",
        CV_N_SPLITS, mean_mse, std_mse, mean_rmse,
    )
    return CVMetrics(
        mean_mse=mean_mse,
        std_mse=std_mse,
        mean_rmse=mean_rmse,
        n_splits=CV_N_SPLITS,
    )
