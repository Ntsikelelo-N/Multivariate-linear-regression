PYTHON     := python3
VENV_DIR   := .venv
PIP        := $(VENV_DIR)/bin/pip
PYTEST     := $(VENV_DIR)/bin/pytest
FLAKE8     := $(VENV_DIR)/bin/flake8
JUPYTER    := $(VENV_DIR)/bin/jupyter
PYTHON_RUN := $(VENV_DIR)/bin/python

.PHONY: install test lint notebook run clean help

## install: Create virtual environment and install dependencies.
install:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Environment ready. Activate with: source $(VENV_DIR)/bin/activate"

## test: Run the pytest suite with verbose output.
test:
	$(PYTEST) tests/ -v --tb=short

## test-cov: Run pytest with HTML coverage report.
test-cov:
	$(PYTEST) tests/ -v --tb=short --cov=src/salary_prediction --cov-report=html
	@echo "✓ Coverage report written to htmlcov/index.html"

## lint: Check code style with flake8 (PEP8 compliance).
lint:
	$(FLAKE8) src/ tests/ config.py --max-line-length=100 --extend-ignore=E203,W503

## notebook: Launch Jupyter for the analysis notebook.
notebook:
	$(JUPYTER) notebook notebooks/01_salary_prediction.ipynb

## run: Execute the end-to-end pipeline (prints evaluation metrics to stdout).
run:
	PYTHONPATH=src $(PYTHON_RUN) -m salary_prediction.pipeline

## clean: Remove generated artefacts, caches, and temporary files.
clean:
	find . -type d -name "__pycache__"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"       -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Clean complete."

## help: List available make targets.
help:
	@grep -E '^##' Makefile | sed 's/## /  /'
