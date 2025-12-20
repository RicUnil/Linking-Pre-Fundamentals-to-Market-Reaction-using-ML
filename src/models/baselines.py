"""
Baseline regression models for excess return prediction.

This module implements simple baseline models to establish
performance benchmarks for testing H₀: excess returns are unpredictable.

Baselines:
1. Dummy mean baseline - predicts the training set mean for all samples
2. CAPM-style baseline - uses per-ticker beta estimated from SPY pre-returns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.config import Settings


logger = logging.getLogger(__name__)


@dataclass
class BaselineResults:
    """
    Container for baseline model predictions and fitted objects.
    
    Attributes
    ----------
    mean_model : DummyRegressor
        Fitted dummy regressor predicting the mean.
    mean_train_pred : np.ndarray
        Mean baseline predictions on training set.
    mean_val_pred : np.ndarray
        Mean baseline predictions on validation set.
    capm_betas : pd.DataFrame
        DataFrame with per-ticker beta and alpha coefficients.
    capm_train_pred : np.ndarray
        CAPM baseline predictions on training set.
    capm_val_pred : np.ndarray
        CAPM baseline predictions on validation set.
    """
    
    mean_model: DummyRegressor
    mean_train_pred: np.ndarray
    mean_val_pred: np.ndarray
    
    capm_betas: pd.DataFrame
    capm_train_pred: np.ndarray
    capm_val_pred: np.ndarray


def train_mean_baseline(
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[DummyRegressor, np.ndarray, np.ndarray]:
    """
    Train a dummy mean regressor baseline.

    The model predicts the mean of y_train for all samples.

    Parameters
    ----------
    y_train : np.ndarray
        Training target values.
    y_val : np.ndarray
        Validation target values.

    Returns
    -------
    model : DummyRegressor
        Fitted dummy regressor.
    y_train_pred : np.ndarray
        Predictions on the training set.
    y_val_pred : np.ndarray
        Predictions on the validation set.
    """
    logger.info("Training mean baseline...")
    
    model = DummyRegressor(strategy="mean")
    # Fit on dummy features (shape doesn't matter for mean strategy)
    model.fit(np.zeros((y_train.shape[0], 1)), y_train)
    
    y_train_pred = model.predict(np.zeros((y_train.shape[0], 1)))
    y_val_pred = model.predict(np.zeros((y_val.shape[0], 1)))
    
    logger.info(f"  Mean baseline: predicts constant {y_train_pred[0]:.6f}")
    
    return model, y_train_pred, y_val_pred


def estimate_capm_betas(
    df_train: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """
    Estimate a simple CAPM beta per ticker using training data.

    For each ticker:
        excess_return_30d ~ alpha + beta * spy_pre_return_30d

    Parameters
    ----------
    df_train : pd.DataFrame
        Cleaned training dataframe with regression target and SPY pre-return.
    settings : Settings
        Project settings, used to access ticker column.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ticker with columns ['beta', 'alpha'].
    """
    logger.info("Estimating CAPM betas per ticker...")
    
    ticker_col = settings.EARNINGS_TICKER_COLUMN
    target_col = "excess_return_30d"
    spy_col = "spy_pre_return_30d"
    
    if spy_col not in df_train.columns:
        raise KeyError(f"Column '{spy_col}' not found in training dataframe.")
    
    betas = []
    
    for ticker, group in df_train.groupby(ticker_col):
        x = group[spy_col].to_numpy()
        y = group[target_col].to_numpy()
        
        # Filter out NaN values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        # Require at least 10 observations
        if x.shape[0] < 10:
            continue
        
        # Simple OLS: y = alpha + beta * x
        x_mean = x.mean()
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).mean()
        var = ((x - x_mean) ** 2).mean()
        
        if var == 0:
            continue
        
        beta = cov / var
        alpha = y_mean - beta * x_mean
        
        betas.append((ticker, beta, alpha))
    
    if not betas:
        logger.warning("No CAPM betas could be estimated; returning empty dataframe.")
        return pd.DataFrame(columns=["beta", "alpha"])
    
    beta_df = pd.DataFrame(betas, columns=[ticker_col, "beta", "alpha"]).set_index(ticker_col)
    
    logger.info(f"  Estimated betas for {len(beta_df)} tickers")
    logger.info(f"  Beta range: [{beta_df['beta'].min():.3f}, {beta_df['beta'].max():.3f}]")
    logger.info(f"  Mean beta: {beta_df['beta'].mean():.3f}")
    
    return beta_df


def predict_capm_baseline(
    df: pd.DataFrame,
    betas: pd.DataFrame,
    settings: Settings,
) -> np.ndarray:
    """
    Generate CAPM-based predictions for a given dataframe using pre-estimated betas.

    For each row:
        y_hat = alpha_ticker + beta_ticker * spy_pre_return_30d

    If a ticker has no beta, fall back to the global mean of the target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe (train or val) containing spy_pre_return_30d and ticker.
    betas : pd.DataFrame
        DataFrame indexed by ticker with columns ['beta', 'alpha'].
    settings : Settings
        Project settings.

    Returns
    -------
    np.ndarray
        Predicted excess returns.
    """
    ticker_col = settings.EARNINGS_TICKER_COLUMN
    spy_col = "spy_pre_return_30d"
    target_col = "excess_return_30d"
    
    preds = np.full(shape=(df.shape[0],), fill_value=np.nan, dtype=float)
    
    # Fallback mean
    global_mean = df[target_col].mean(skipna=True)
    
    for i, row in df.iterrows():
        ticker = row[ticker_col]
        spy_r = row.get(spy_col, np.nan)
        
        # Get position in dataframe
        idx = df.index.get_loc(i)
        
        if pd.isna(spy_r):
            preds[idx] = global_mean
            continue
        
        if ticker in betas.index:
            beta = betas.loc[ticker, "beta"]
            alpha = betas.loc[ticker, "alpha"]
            preds[idx] = alpha + beta * spy_r
        else:
            preds[idx] = global_mean
    
    return preds


def train_baseline_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    settings: Settings,
) -> BaselineResults:
    """
    Train both mean and CAPM-style baselines and return predictions.

    Parameters
    ----------
    df_train : pd.DataFrame
        Cleaned training dataframe.
    df_val : pd.DataFrame
        Cleaned validation dataframe.
    y_train : np.ndarray
        Training target (excess_return_30d).
    y_val : np.ndarray
        Validation target.
    settings : Settings
        Project settings.

    Returns
    -------
    BaselineResults
        Container with fitted baselines and predictions.
    """
    logger.info("=" * 70)
    logger.info("TRAINING BASELINE MODELS")
    logger.info("=" * 70)
    
    # Mean baseline
    mean_model, mean_train_pred, mean_val_pred = train_mean_baseline(y_train, y_val)
    
    # CAPM baseline
    betas = estimate_capm_betas(df_train, settings)
    
    logger.info("\nGenerating CAPM predictions...")
    capm_train_pred = predict_capm_baseline(df_train, betas, settings)
    capm_val_pred = predict_capm_baseline(df_val, betas, settings)
    
    logger.info(f"  Train predictions: {np.isfinite(capm_train_pred).sum()} / {len(capm_train_pred)}")
    logger.info(f"  Val predictions: {np.isfinite(capm_val_pred).sum()} / {len(capm_val_pred)}")
    
    return BaselineResults(
        mean_model=mean_model,
        mean_train_pred=mean_train_pred,
        mean_val_pred=mean_val_pred,
        capm_betas=betas,
        capm_train_pred=capm_train_pred,
        capm_val_pred=capm_val_pred,
    )
