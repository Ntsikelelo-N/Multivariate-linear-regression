# Employee Salary Prediction: Multivariate Linear Regression

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-lightgrey)

> **What the model does:** Predict annual employee salary from field-of-work and an
> engineered *experience-value* composite, using OLS regression validated with
> VIF analysis, Shapiro-Wilk normality testing, and 20-fold cross-validation.

---

## Table of Contents

- [Business Problem](#business-problem)
- [Project Architecture](#project-architecture)
- [Key Results](#key-results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Limitations & Future Work](#limitations--future-work)

---

## Business Problem

Organisations often need to benchmark salaries against internal equity factors
(experience, seniority) and market data. This project builds an interpretable
regression model that quantifies how much each factor contributes to annual
salary, giving HR teams a defensible, data-driven reference point.

---

## Project Architecture

```
Raw CSV
   │
   ▼
data_loader.py ──► preprocessing.py ──► features.py
                         │                   │
                    Clean DataFrame    Feature Matrix (X)
                                            │
                                       model.py
                                   (OLS + k-fold CV)
                                            │
                                  Coefficients + Metrics
                                            │
                               visualization.py (EDA + diagnostics)
```

Each stage is a pure-function module in `src/salary_prediction/` and is
independently testable. The notebook in `notebooks/` stitches them together
and documents the reasoning.

---

## Key Results

| Metric | Train | Test (80/20) | 20-Fold CV |
|--------|-------|-------------|------------|
| R²     | ~0.63 | ~0.62       | ~0.61      |
| MAE    | —     | ~$6 125     | —          |
| RMSE   | —     | ~$7 783     | ~$7 916    |

**Top predictors (by coefficient magnitude):**

| Feature             | Coefficient   | Interpretation                         |
|---------------------|--------------|----------------------------------------|
| Experience Value    | +$494        | Per unit increase in composite score   |
| Field: Engineering  | +$12 800     | Premium over Human Resources baseline  |
| Field: Finance      | +$9 693      | Premium over Human Resources baseline  |
| Field: Marketing    | +$4 020      | Premium over Human Resources baseline  |
| Intercept           | ~$35 250     | Minimum predicted salary               |

---

## Quick Start

### 1. Clone and enter the repo

```bash
git clone https://github.com/Ntsikelelo-N/Multivariate-linear-regression.git
cd Multivariate-linear-regression
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4a. Run the notebook (exploratory)

```bash
jupyter notebook notebooks/01_salary_prediction.ipynb
```

### 4b. Run the pipeline as a script

```bash
python -m salary_prediction.pipeline 
```

### 5. Run tests

```bash
pytest tests/ -v
```

Or use the Makefile:

```bash
make install   
make test      
make notebook  
make clean     
```

---

## Project Structure

```
.
├── data/
│   └── salary.csv              
├── notebooks/
│   └── 01_salary_prediction.ipynb   
├── src/
│   └── salary_prediction/
│       ├── __init__.py
│       ├── data_loader.py      
│       ├── preprocessing.py    
│       ├── features.py         
│       ├── model.py            
│       ├── pipeline.py       
│       └── visualization.py    
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_model.py
├── .gitignore
├── config.py                   
├── Makefile
├── README.md
└── requirements.txt
```

---

## Methodology

### 1. Data Cleaning
- Decoded integer-coded categoricals (position, field, gender) to readable labels.
- Dropped `yearsabs` (days/weeks absent) after finding the column unreliable:
  values exceeding plausible lifetime-absent periods and long-tenured employees
  with zero days absent.
- Imputed the single missing salary using the employee's own market-rate
  multiplier applied to the mean salary for executives in marketing.

### 2. Feature Selection
- Pearson correlation + p-value filtering (|r| > 0.30, p < 0.05).
- VIF analysis to detect multicollinearity.  Features with VIF > 5 were either
  dropped or consolidated.
- `Experience Value = (yearsworked + yearsrank) × market` combines three
  collinear predictors into a single, interpretable composite.

### 3. Encoding
- One-hot encoding for `Field` (reference: Human Resources).
- Ordinal encoding for `Position` (Junior < Manager < Executive).
- `Has degree` / `Has other qualification` dropped due to severe class imbalance.

### 4. Regression & Validation
- OLS via `statsmodels` (preserves p-values, confidence intervals, and
  summary diagnostics).
- Residual normality checked with Shapiro-Wilk; mild non-normality tolerated
  because OLS is robust at n > 500.
- Homoscedasticity assessed via residuals-vs-fitted plot.
- 20-fold cross-validation to confirm generalisation.

---

## Limitations & Future Work

- **Dataset size:** 514 rows limits the reliability of some subgroup analyses
  (e.g., non-degree holders, n = 18).
- **Normality:** Shapiro-Wilk rejects H₀ - consider Box-Cox transformation or
  a log-salary target.
- **Missing signal:** 38% of variance is unexplained; performance, bonuses, and
  education quality are not captured.
- **Next steps:**
  - Ridge / Lasso to regularise and auto-select features.
  - Tree-based models (Random Forest, XGBoost) to capture non-linear effects.
  - Deploy as a FastAPI endpoint for HR self-service queries.
  - Track experiments with MLflow.

---

## Data Source

Dataset: `salary.csv` 

Column metadata reference:
[Salary metadata CSV](https://github.com/PhumlaniKubeka/StatisticalThinking/blob/master/Salary%20metadata.csv)

---
