# Employee Salary Prediction using Multivariate Linear Regression

## Project Overview
This notebook demonstrates how to use **multivariate linear regression** to predict employee salaries based on personal and professional attributes. It focuses on exploring feature relationships, validating model assumptions, and evaluating prediction accuracy using statistical and machine learning tools.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Tools and Libraries](#tools-and-libraries)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)

## Introduction
The notebook explores how combinations of features explain variations in employee salaries. It includes:
- Exploratory data analysis and visualization
- Detection of missing/extreme values
- Evaluation of linear regression assumptions
- Feature transformation where needed
- Model training and performance evaluation

## Dataset
The main dataset is `salary.csv`, containing employee attributes and their corresponding salaries.

### Key Features:
- `Years worked in current field`, `Years worked at current rank`, `Position` and `Field of work`.
- Target: `Annual salary`

Metadata reference: [Salary metadata CSV](https://github.com/PhumlaniKubeka/StatisticalThinking/blob/master/Salary%20metadata.csv)

## Methodology
1. **Data Preprocessing**: Handled missing values, visualized distributions, applied encoding.
2. **Exploratory Analysis**: Used correlation plots, scatterplots, histograms.
3. **Model Assumptions**: Checked multicollinearity with VIF, linearity, normality, homoscedasticity.
4. **Regression Modeling**:
    - Fitted a multivariate OLS regression model using `statsmodels`.
    - Calculated p-values and R² for feature evaluation.
5. **Evaluation**:
    - Split data into training and test sets.
    - Assessed using RMSE, MAE, and adjusted R².

## Tools and Libraries
- `pandas`, `numpy`
- `seaborn`, `matplotlib`
- `statsmodels`, `scikit-learn`
- `scipy.stats` for statistical testing

## Model Evaluation
- **Cross-validation**: Performed K-Fold cross-validation.
- **Metrics**:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**

## Results
- Key predictors of salary included **education level**, **years of experience**, and **seniority**.
- Model performance metrics showed reasonable generalization with some variance explained.

## Conclusion
Multivariate linear regression is a powerful tool for understanding how various factors influence employee salary. While the model showed good explanatory power, future improvements could further increase prediction accuracy.

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/Ntsikelelo-N/Multivariate-linear-regression.git
    cd Ntsikelelo-N/Multivariate-linear-regression
    ```
2. Ensure the dataset is available in the `data/` directory as `salary.csv`.
3. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
4. Open the notebook:
    ```bash
    jupyter notebook multivariate_linear_regression.ipynb
    ```

## Future Improvements
- Apply regularized regression (Ridge/Lasso) for feature selection.
- Experiment with tree-based models for non-linear effects.
