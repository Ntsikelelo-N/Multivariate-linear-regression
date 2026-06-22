"""
salary_prediction — OLS-based employee salary prediction pipeline.

Public API
----------
from salary_prediction import load_data, preprocess, engineer_features, train_model
"""

__version__ = "0.1.0"
__author__ = "Ntsikelelo Nicholas Jantjie"

from .data_loader import load_data
from .preprocessing import preprocess
from .features import engineer_features
from .model import train_model
