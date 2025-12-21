# Project Proposal: Linking Pre-Fundamentals to Market Returns with Machine Learning

**Advanced Programming 2025 — HEC Lausanne**  
**Ricardo Contente Guerreiro | MSc Finance**

---

## Research Question

Can we predict post-earnings announcement excess returns by linking pre-announcement fundamental data with subsequent market performance using machine learning?

## Motivation

Earnings announcements represent critical information events where companies disclose their financial performance. The central question in financial economics is whether fundamental data released before and during these announcements can predict how the market will react in the following 30 days. This project uses machine learning to systematically test whether pre-announcement fundamentals (earnings surprises, profitability metrics, valuation ratios) contain predictive information about post-announcement excess returns relative to the S&P 500 benchmark.

## What We'll Build

We will construct a complete machine learning pipeline that:

1. **Links fundamentals to market outcomes**: Extract pre-announcement fundamental data (earnings surprises, revenue growth, profitability ratios, valuation metrics) and connect them to post-announcement market returns over a 30-day window.

2. **Engineers predictive features**: Transform raw fundamental data into 21 engineered features spanning fundamental metrics (P/E ratios, ROE, leverage), market indicators (pre-announcement returns, volatility), and momentum signals (1-month, 3-month, 6-month returns).

3. **Trains multiple ML models**: Implement both regression (Ridge, Random Forest, XGBoost) and classification (Logistic Regression, Gradient Boosting) approaches to predict continuous returns and directional movements.

4. **Validates rigorously**: Use temporal train/validation/test splits to prevent look-ahead bias, conduct 5-fold cross-validation, and perform statistical hypothesis testing (bootstrap confidence intervals, permutation tests).

## Data Sources

- **Earnings data**: Capital IQ-style fundamental data covering S&P 500 companies (2015-2024)
- **Price data**: Daily stock prices and SPY benchmark returns from Yahoo Finance
- **Target variable**: 30-day excess return = (Stock return) - (SPY return) over the 30 days following earnings announcement

The dataset comprises approximately 19,300 earnings announcements with 21 engineered features per observation.

## Methodology

We approach this as both a **regression problem** (predicting continuous excess returns) and a **classification problem** (predicting whether a stock will outperform or underperform the market). The pipeline includes:

- Temporal data splits to simulate real-world deployment
- Feature engineering from fundamental and market data
- Multiple baseline and ML models for comparison
- Comprehensive evaluation using R², MAE, RMSE (regression) and ROC-AUC, accuracy (classification)
- Statistical testing to validate findings beyond simple metrics

## Expected Outcome

This project will provide empirical evidence on whether pre-announcement fundamentals can predict post-announcement market reactions. If successful, it demonstrates that fundamental analysis has predictive power. If unsuccessful (R² ≈ 0, AUC ≈ 0.50), it provides evidence for market efficiency—that fundamental information is already priced in by the time of the announcement.

Either outcome contributes valuable insights: positive results suggest exploitable patterns, while negative results support the Efficient Market Hypothesis for liquid, well-covered stocks.

---
 
**Course:** Advanced Programming 2025 — HEC Lausanne  
**December 2025**

