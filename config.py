"""
config.py — centralised configuration for the salary-prediction pipeline.

CHANGED: Extracted every literal constant that appeared inside notebook cells
(split ratios, random seeds, thresholds, column mappings) into named constants
here. This eliminates magic numbers, makes assumptions explicit, and lets
reviewers adjust behaviour without hunting through notebook code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "salary.csv"

# ---------------------------------------------------------------------------
# Random state — set once, used everywhere
# ---------------------------------------------------------------------------
RANDOM_STATE = 2025

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20          # 80 / 20 split

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_N_SPLITS = 20

# ---------------------------------------------------------------------------
# Feature selection thresholds
# ---------------------------------------------------------------------------
# Minimum |Pearson r| for a feature to be considered a candidate predictor
CORRELATION_THRESHOLD = 0.30

# Maximum Pearson r between *non-target* features before flagging collinearity
INTER_FEATURE_CORRELATION_THRESHOLD = 0.40

# p-value cutoff for statistical significance
P_VALUE_ALPHA = 0.05

# Variance Inflation Factor ceiling; features above this are consolidated
VIF_THRESHOLD = 5.0

# Shapiro-Wilk significance level for residual normality check
NORMALITY_ALPHA = 0.05

# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------
# The single missing salary belongs to an Executive in Marketing.
# Their recorded market value (0.93) is used to scale the group mean salary,
# producing a market-adjusted estimate rather than a flat group mean.
# The value 0.93 is read from the dataset row itself — see preprocessing.py.
IMPUTATION_POSITION = "Executive"
IMPUTATION_FIELD    = "Marketing"

# ---------------------------------------------------------------------------
# Column name mappings (raw CSV → cleaned labels)
# ---------------------------------------------------------------------------
COLUMN_RENAME_MAP = {
    "salary":       "Annual salary",
    "exprior":      "Years of experience in prior field",
    "yearsworked":  "Years worked in current field",
    "yearsrank":    "Years worked at current rank",
    "market":       "Market value",
    "degree":       "Has degree",
    "otherqual":    "Has other post-secondary qualification",
    "position":     "Position",
    "male":         "Gender",
    "Field":        "Field",
    "yearsabs":     "Years absent from work",
}

DEGREE_MAP   = {0: "No", 1: "Yes"}
OTHERQUAL_MAP = {0: "No", 1: "Yes"}
GENDER_MAP   = {0: "Female", 1: "Male"}
POSITION_MAP = {1: "Junior employee", 2: "Manager", 3: "Executive"}
FIELD_MAP    = {1: "Engineering", 2: "Finance", 3: "Human resources", 4: "Marketing"}

POSITION_ORDER = ["Junior employee", "Manager", "Executive"]
FIELD_ORDER    = ["Human resources", "Engineering", "Finance", "Marketing"]

# ---------------------------------------------------------------------------
# Modelling — feature names
# ---------------------------------------------------------------------------
# Engineered composite: (yearsworked + yearsrank) × market
EXPERIENCE_VALUE_COL = "Experience value"

TARGET_COL = "Annual salary"

MODEL_FEATURES = [
    EXPERIENCE_VALUE_COL,
    "Field_Engineering",
    "Field_Finance",
    "Field_Marketing",
]

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
FIGURE_SIZE         = (10, 6)
HISTOGRAM_BINS      = 10
SCATTER_ALPHA       = 0.6
PLOT_TITLE_FONTSIZE = 14
PLOT_LABEL_FONTSIZE = 12
