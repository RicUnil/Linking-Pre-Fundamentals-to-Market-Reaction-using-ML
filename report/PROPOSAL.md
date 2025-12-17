# Project Proposal — Advanced Programming 2025

## 1. Project Title

**Predicting Post-Earnings Announcement Excess Returns: A Machine Learning Approach**

## 2. Motivation and Context

Earnings announcements are among the most significant events in financial markets, often triggering substantial price movements. Understanding whether these movements are predictable can provide insights into market efficiency and inform trading strategies.

This project investigates whether machine learning models can predict the 30-day excess return of stocks following earnings announcements, relative to the SPY (S&P 500 ETF) benchmark.

## 3. Research Question and Null Hypothesis

**Research Question:**  
Can we predict post-earnings announcement excess returns using fundamental financial data and market indicators?

**Null Hypothesis (H₀):**  
Post-earnings excess returns (relative to SPY) are unpredictable and follow a random distribution.

**Alternative Hypothesis (H₁):**  
Post-earnings excess returns exhibit predictable patterns that can be captured by machine learning models.

## 4. Data Description

### Primary Datasets

- **RAW_DATA.csv** — Main earnings dataset containing:
  - Company identifiers
  - Earnings announcement dates
  - Financial fundamentals (Capital IQ style)
  - Historical price data

- **BENCHMARK.csv** — SPY benchmark data:
  - Quarterly SPY returns
  - Market performance metrics

- **Quarter_1.csv – Quarter_4.csv** — Quarterly supplementary data:
  - Additional fundamental metrics
  - Sector classifications
  - Market conditions

### Target Variable

- **30-day excess return** = (Stock return over 30 days post-earnings) - (SPY return over same period)
- **Binary label** = 1 if excess return > 0 (outperform), 0 otherwise (underperform)

## 5. Methodology Overview

### 5.1 Data Pipeline

1. **Data Loading and Validation** — Import and validate all datasets
2. **Data Cleaning** — Handle missing values, outliers, and inconsistencies
3. **Feature Engineering** — Create predictive features from fundamentals and market data
4. **Target Construction** — Calculate 30-day excess returns

### 5.2 Feature Categories

- **Fundamental Features:** EPS surprise, revenue growth, profit margins, leverage ratios
- **Market Features:** Pre-announcement volatility, trading volume, momentum indicators
- **Temporal Features:** Quarter, year, days since last earnings
- **Sector Features:** Industry classification, sector performance

### 5.3 Modeling Approach

**Baseline Models:**
- Historical mean excess return
- CAPM-based prediction
- Naive classifier (majority class)

**Machine Learning Models:**
- Linear Regression (with regularization)
- Random Forest (regression and classification)
- Gradient Boosting (XGBoost/LightGBM)
- Logistic Regression (for classification)

### 5.4 Evaluation Strategy

- **Temporal Train/Test Split:** Data before 2020-01-01 for training, after for testing
- **Rolling Window Validation:** Simulate real-world deployment
- **Regression Metrics:** RMSE, MAE, R²
- **Classification Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Statistical Tests:** t-tests, permutation tests for H₀

## 6. Expected Contributions and Limitations

### Expected Contributions

- Rigorous test of market efficiency in the post-earnings context
- Identification of most predictive features for excess returns
- Comparison of ML approaches vs. traditional financial models
- Reproducible, modular pipeline for financial ML research

### Limitations

- Limited to available fundamental data (no alternative data sources)
- Survivorship bias in historical data
- Transaction costs and market impact not modeled
- Model performance may degrade in regime changes

## 7. Timeline

| Week | Tasks |
|------|-------|
| 1-2  | Data loading, cleaning, and exploratory analysis |
| 3-4  | Feature engineering and target construction |
| 5-6  | Baseline model implementation and evaluation |
| 7-8  | ML model development and hyperparameter tuning |
| 9-10 | Out-of-sample testing and hypothesis testing |
| 11-12| Final report writing and presentation preparation |

## 8. Technical Requirements

- **Language:** Python 3.10+
- **Key Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Code Standards:** PEP8, type hints, NumPy docstrings
- **Testing:** pytest for unit tests
- **Reproducibility:** Fixed random seed (42), version control

## 9. Deliverables

1. Complete 20-step modular pipeline
2. Comprehensive test suite
3. Final report with findings and visualizations
4. Presentation slides
5. Reproducible code repository

---

**Project Start Date:** December 2025  
**Course:** Advanced Programming 2025 — HEC Lausanne
