import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CV_N_SPLITS, MODEL_FEATURES, NORMALITY_ALPHA, TARGET_COL
from src.salary_prediction.model import (
    CVMetrics,
    HoldoutMetrics,
    ModelResult,
    _check_normality,
    _cross_validate,
    _evaluate_holdout,
    _fit_ols,
    _split,
    train_model,
)


@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate a small synthetic feature matrix for fast tests.

    Using a deterministic linear relationship ensures holdout R² is high
    enough that metric-range assertions (R² > 0) are reliable.
    """
    np.random.seed(2025)
    n = 200
    X = pd.DataFrame(
        {
            "Experience value": np.random.uniform(0, 50, n),
            "Field_Engineering": np.random.randint(0, 2, n),
            "Field_Finance": np.random.randint(0, 2, n),
            "Field_Marketing": np.random.randint(0, 2, n),
        }
    )
    # True salary = 35_000 + 500*ev + 12_000*eng + noise
    y = pd.Series(
        35_000
        + 500 * X["Experience value"]
        + 12_000 * X["Field_Engineering"]
        + np.random.normal(0, 3_000, n),
        name=TARGET_COL,
    )
    return X, y


class TestSplit:
    def test_split_sizes(self, synthetic_data):
        X, y = synthetic_data
        X_tr, X_te, y_tr, y_te = _split(X, y)
        assert len(X_tr) + len(X_te) == len(X)
        assert len(y_tr) + len(y_te) == len(y)

    def test_test_fraction_is_approximately_20_percent(self, synthetic_data):
        X, y = synthetic_data
        _, X_te, _, _ = _split(X, y)
        assert abs(len(X_te) / len(X) - 0.20) < 0.02


class TestFitOLS:
    def test_returns_fitted_model(self, synthetic_data):
        import statsmodels.api as sm

        X, y = synthetic_data
        X_tr, _, y_tr, _ = _split(X, y)
        model = _fit_ols(X_tr, y_tr)
        assert hasattr(model, "rsquared")

    def test_r_squared_is_between_0_and_1(self, synthetic_data):
        import statsmodels.api as sm

        X, y = synthetic_data
        X_tr, _, y_tr, _ = _split(X, y)
        model = _fit_ols(X_tr, y_tr)
        assert 0.0 <= model.rsquared <= 1.0


class TestCheckNormality:
    def test_truly_normal_residuals_pass(self):
        np.random.seed(0)
        residuals = pd.Series(np.random.normal(0, 1, 500))
        passed, p_val = _check_normality(residuals, alpha=NORMALITY_ALPHA)
        assert passed == True
        assert p_val > NORMALITY_ALPHA

    def test_uniform_residuals_fail(self):
        np.random.seed(0)
        residuals = pd.Series(np.random.uniform(-10, 10, 500))
        passed, p_val = _check_normality(residuals, alpha=NORMALITY_ALPHA)
        assert passed == False

    def test_returns_float_p_value(self):
        residuals = pd.Series(np.random.normal(0, 1, 100))
        _, p_val = _check_normality(residuals)
        assert isinstance(p_val, float)
        assert 0.0 <= p_val <= 1.0


class TestEvaluateHoldout:
    def test_metrics_are_non_negative(self, synthetic_data):
        import statsmodels.api as sm

        X, y = synthetic_data
        X_tr, X_te, y_tr, y_te = _split(X, y)
        model = _fit_ols(X_tr, y_tr)
        metrics = _evaluate_holdout(model, X_te, y_te)
        assert metrics.mae >= 0
        assert metrics.rmse >= 0

    def test_rmse_gte_mae(self, synthetic_data):
        """RMSE ≥ MAE is a mathematical property, not a coincidence."""
        import statsmodels.api as sm

        X, y = synthetic_data
        X_tr, X_te, y_tr, y_te = _split(X, y)
        model = _fit_ols(X_tr, y_tr)
        metrics = _evaluate_holdout(model, X_te, y_te)
        assert metrics.rmse >= metrics.mae

    def test_r2_bounded(self, synthetic_data):
        import statsmodels.api as sm

        X, y = synthetic_data
        X_tr, X_te, y_tr, y_te = _split(X, y)
        model = _fit_ols(X_tr, y_tr)
        metrics = _evaluate_holdout(model, X_te, y_te)
        assert metrics.r2 > 0


class TestCrossValidate:
    def test_n_splits_matches_config(self, synthetic_data):
        X, y = synthetic_data
        cv = _cross_validate(X, y)
        assert cv.n_splits == CV_N_SPLITS

    def test_mean_mse_is_positive(self, synthetic_data):
        X, y = synthetic_data
        cv = _cross_validate(X, y)
        assert cv.mean_mse > 0

    def test_mean_rmse_equals_sqrt_mean_mse(self, synthetic_data):
        X, y = synthetic_data
        cv = _cross_validate(X, y)
        assert abs(cv.mean_rmse - np.sqrt(cv.mean_mse)) < 1.0


class TestTrainModel:
    def test_returns_model_result(self, synthetic_data):
        X, y = synthetic_data
        result = train_model(X, y)
        assert isinstance(result, ModelResult)

    def test_holdout_is_populated(self, synthetic_data):
        X, y = synthetic_data
        result = train_model(X, y)
        assert isinstance(result.holdout, HoldoutMetrics)

    def test_cv_is_populated(self, synthetic_data):
        X, y = synthetic_data
        result = train_model(X, y)
        assert isinstance(result.cv, CVMetrics)

    def test_y_pred_length_matches_test_set(self, synthetic_data):
        X, y = synthetic_data
        result = train_model(X, y)
        assert len(result.y_pred) == len(result.y_test)

    def test_normality_flag_is_bool(self, synthetic_data):
        X, y = synthetic_data
        result = train_model(X, y)
        assert isinstance(result.normality_pass, bool)
