"""
visualization.py — EDA and model diagnostic plots.

CHANGED: The original ``relationship_graph`` function silently captured the
global ``cleaned_df`` variable, meaning it would produce incorrect plots if
called after any mutation of that global. Every function here accepts an
explicit DataFrame argument, making plots reproducible and testable.

Usage (notebook)
----------------
from salary_prediction.visualization import (
    plot_salary_distribution,
    plot_correlation_heatmap,
    plot_residuals_vs_fitted,
    plot_residual_histogram,
    plot_actual_vs_predicted,
)
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    FIGURE_SIZE,
    HISTOGRAM_BINS,
    PLOT_LABEL_FONTSIZE,
    PLOT_TITLE_FONTSIZE,
    SCATTER_ALPHA,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_salary_by_feature(
    df: pd.DataFrame,
    feature: str,
    x_label: str | None = None,
) -> plt.Figure:
    """Plot salary distribution broken down by a single feature.

    Categorical features → overlaid KDE histogram by group.
    Numeric features     → scatter plot against annual salary.

    Parameters
    ----------
    df:
        Cleaned DataFrame (output of preprocessing).
    feature:
        Column name to plot against ``Annual salary``.
    x_label:
        Optional axis label; defaults to ``feature``.

    Returns
    -------
    matplotlib.figure.Figure
        Caller is responsible for calling ``plt.show()`` or saving.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    label = x_label or feature

    if pd.api.types.is_object_dtype(df[feature]) or pd.api.types.is_categorical_dtype(df[feature]):
        sns.histplot(
            data=df,
            x=TARGET_COL,
            hue=feature,
            kde=True,
            bins=HISTOGRAM_BINS,
            edgecolor="black",
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(
            f"Annual salary distribution by {feature}",
            fontweight="bold",
            fontsize=PLOT_TITLE_FONTSIZE,
        )
        ax.set_xlabel("Annual salary (USD)", fontweight="bold", fontsize=PLOT_LABEL_FONTSIZE)
        ax.set_ylabel("Frequency", fontweight="bold", fontsize=PLOT_LABEL_FONTSIZE)
        ax.tick_params(axis="x", rotation=45)
    else:
        sns.scatterplot(
            data=df,
            x=feature,
            y=TARGET_COL,
            alpha=SCATTER_ALPHA,
            ax=ax,
        )
        ax.set_title(
            f"Annual salary vs {feature}",
            fontweight="bold",
            fontsize=PLOT_TITLE_FONTSIZE,
        )
        ax.set_xlabel(label, fontweight="bold", fontsize=PLOT_LABEL_FONTSIZE)
        ax.set_ylabel("Annual salary (USD)", fontweight="bold", fontsize=PLOT_LABEL_FONTSIZE)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Pearson correlation heatmap for all numeric columns.

    Parameters
    ----------
    df:
        Encoded DataFrame (post-preprocessing).

    Returns
    -------
    matplotlib.figure.Figure
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix", fontweight="bold", fontsize=PLOT_TITLE_FONTSIZE)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_residual_histogram(residuals: pd.Series) -> plt.Figure:
    """Histogram of OLS residuals to visually assess normality.

    Parameters
    ----------
    residuals:
        ``model.resid`` from a fitted statsmodels OLS result.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(residuals.mean(), color="red", linestyle="--", label=f"Mean = {residuals.mean():.1f}")
    ax.set_title("Histogram of Residuals", fontweight="bold", fontsize=PLOT_TITLE_FONTSIZE)
    ax.set_xlabel("Residuals", fontweight="bold", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("Frequency", fontweight="bold", fontsize=PLOT_LABEL_FONTSIZE)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals_vs_fitted(
    fitted_values: pd.Series,
    residuals: pd.Series,
) -> plt.Figure:
    """Scatter of residuals against fitted values to assess homoscedasticity.

    A random, horizontal band centred on zero indicates constant variance.
    Funnel shapes or curves suggest heteroscedasticity or model mis-specification.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.scatter(fitted_values, residuals, alpha=SCATTER_ALPHA, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    # LOESS smoothing line to make any trend visible
    try:
        lowess = sm.nonparametric.lowess(residuals, fitted_values, frac=0.4)
        ax.plot(lowess[:, 0], lowess[:, 1], color="orange", linewidth=2, label="LOESS")
        ax.legend()
    except Exception:
        pass  # LOESS is cosmetic; don't fail the whole diagnostic

    ax.set_title("Residuals vs Fitted Values", fontweight="bold", fontsize=PLOT_TITLE_FONTSIZE)
    ax.set_xlabel("Fitted Values", fontweight="bold", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("Residuals", fontweight="bold", fontsize=PLOT_LABEL_FONTSIZE)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: pd.Series,
) -> plt.Figure:
    """Scatter of actual vs predicted salaries on the hold-out test set.

    A perfect model would place all points on the 45° diagonal.

    Parameters
    ----------
    y_test:
        Actual salary values from the test split.
    y_pred:
        Model predictions for the test split.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.scatter(y_test, y_pred, alpha=SCATTER_ALPHA, color="steelblue", label="Predictions")

    # Perfect-prediction diagonal
    lims = [
        min(y_test.min(), y_pred.min()),
        max(y_test.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")

    ax.set_title("Actual vs Predicted Salary (Test Set)", fontweight="bold", fontsize=PLOT_TITLE_FONTSIZE)
    ax.set_xlabel("Actual Salary (USD)", fontweight="bold", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("Predicted Salary (USD)", fontweight="bold", fontsize=PLOT_LABEL_FONTSIZE)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_qq(residuals: pd.Series) -> plt.Figure:
    """Q-Q plot of residuals against a normal distribution.

    Points hugging the diagonal indicate normality; heavy tails indicate
    skew or outliers. Complements the Shapiro-Wilk test with a visual check.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    sm.qqplot(residuals, line="s", ax=ax, alpha=SCATTER_ALPHA)
    ax.set_title("Q-Q Plot of Residuals", fontweight="bold", fontsize=PLOT_TITLE_FONTSIZE)
    fig.tight_layout()
    return fig
