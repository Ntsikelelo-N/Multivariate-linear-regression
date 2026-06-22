"""
pipeline.py — end-to-end execution entry point.

Run as a module:
    python -m salary_prediction.pipeline

Or import and call programmatically:
    from salary_prediction.pipeline import run
    result = run()

CHANGED: New file. Without this, the only way to run the project was to
execute every notebook cell in sequence. This script makes the pipeline
runnable in CI, cron jobs, or any non-notebook environment.
"""

import logging
import sys
from pathlib import Path

# Resolve config.py from the project root (two levels above src/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_PATH

from salary_prediction.data_loader import load_data
from salary_prediction.preprocessing import preprocess
from salary_prediction.features import engineer_features
from salary_prediction.model import train_model, print_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(data_path: Path = RAW_DATA_PATH):
    """Execute the full pipeline and return a :class:`ModelResult`.

    Parameters
    ----------
    data_path:
        Path to the raw salary CSV. Defaults to ``config.RAW_DATA_PATH``.

    Returns
    -------
    ModelResult
        Fitted model plus evaluation metrics.
    """
    logger.info("=== Salary Prediction Pipeline START ===")

    raw_df = load_data(data_path)
    clean_df = preprocess(raw_df)
    X, y = engineer_features(clean_df)
    result = train_model(X, y)

    print_summary(result)

    logger.info("=== Pipeline COMPLETE ===")
    return result


if __name__ == "__main__":
    run()
