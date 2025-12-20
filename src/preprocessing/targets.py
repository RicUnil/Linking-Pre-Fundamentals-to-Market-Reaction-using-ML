"""
Target construction utilities for post-earnings returns.

This module provides functions to compute 30-day post-earnings returns,
excess returns vs SPY, and binary outperformance labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import Settings


logger = logging.getLogger(__name__)


def _select_price_column(daily_df: pd.DataFrame) -> str:
    """
    Choose which price column to use for return computation.
    
    Preference order:
    1. 'adj_close' (adjusted close - preferred for accurate returns)
    2. 'close' (regular close - fallback)
    
    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily OHLCV data for a single ticker.
        
    Returns
    -------
    str
        Name of the selected price column.
        
    Raises
    ------
    ValueError
        If neither 'adj_close' nor 'close' is present.
        
    Examples
    --------
    >>> df = pd.DataFrame({'adj_close': [100, 101], 'close': [100, 101]})
    >>> _select_price_column(df)
    'adj_close'
    """
    if 'adj_close' in daily_df.columns:
        return 'adj_close'
    elif 'close' in daily_df.columns:
        return 'close'
    else:
        raise ValueError(
            f"Neither 'adj_close' nor 'close' found in columns. "
            f"Available columns: {list(daily_df.columns)}"
        )


def compute_window_return(
    daily_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Optional[float]:
    """
    Compute the simple total return over [start_date, end_date] for a single ticker.
    
    The return is defined as:
        (price_end / price_start) - 1
    
    This represents the total return over the window, assuming buy-and-hold.
    
    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily OHLCV data indexed by datetime.
    start_date : pd.Timestamp
        Start date of the window (inclusive).
    end_date : pd.Timestamp
        End date of the window (inclusive).
        
    Returns
    -------
    Optional[float]
        The total return over the window, or None if data is insufficient.
        Returns None if:
        - Fewer than 2 data points in the window
        - Price data is missing or invalid
        - Start or end prices are zero or negative
        
    Examples
    --------
    >>> dates = pd.date_range("2020-01-01", periods=3, freq="D")
    >>> df = pd.DataFrame({"adj_close": [100.0, 110.0, 121.0]}, index=dates)
    >>> compute_window_return(df, dates[0], dates[-1])
    0.21
    """
    # Ensure index is datetime
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df = daily_df.copy()
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df.sort_index()
    
    # Filter to window
    mask = (daily_df.index >= start_date) & (daily_df.index <= end_date)
    window_df = daily_df.loc[mask]
    
    # Need at least 2 data points
    if len(window_df) < 2:
        return None
    
    # Select price column
    try:
        price_col = _select_price_column(window_df)
    except ValueError:
        return None
    
    # Get first and last prices
    price_start = window_df[price_col].iloc[0]
    price_end = window_df[price_col].iloc[-1]
    
    # Check for valid prices
    if pd.isna(price_start) or pd.isna(price_end):
        return None
    
    if price_start <= 0 or price_end <= 0:
        return None
    
    # Compute simple return
    return_value = (price_end / price_start) - 1.0
    
    return float(return_value)


def add_post_earnings_targets(
    earnings_with_windows: pd.DataFrame,
    settings: Settings,
    daily_data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    For each earnings event, compute 30-day post-earnings returns and labels.
    
    This function computes:
    - stock_return_30d: Stock's 30-day return after earnings
    - spy_return_30d: SPY's 30-day return over the same window
    - excess_return_30d: Stock return minus SPY return (alpha)
    - label_outperform_30d: Binary label (1 if excess > 0, else 0)
    
    Only events with has_full_post_window=True are processed. Others get NaN.
    
    Parameters
    ----------
    earnings_with_windows : pd.DataFrame
        Earnings dataset with window columns and coverage flags from Step 06.
        Must contain: ticker column, date column, post_window_start,
        post_window_end, has_full_post_window.
    settings : Settings
        Global project settings.
    daily_data_dir : Path, optional
        Directory containing '{TICKER}_daily.parquet' files.
        Default is results/step_05/.
        
    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with additional columns:
        - stock_return_30d: float
        - spy_return_30d: float
        - excess_return_30d: float
        - label_outperform_30d: int (0 or 1)
        
    Notes
    -----
    - Returns are computed using adjusted close prices when available
    - Events without full post-window coverage get NaN values
    - Missing ticker files are logged as warnings
    - Summary statistics are logged at completion
    """
    logger.info("Computing 30-day post-earnings targets...")
    
    # Set default daily data directory
    if daily_data_dir is None:
        daily_data_dir = settings.RESULTS_DIR / "step_05"
    
    logger.info(f"  Daily data directory: {daily_data_dir}")
    
    # Try to load consolidated data first
    consolidated_path = daily_data_dir / "cache" / "all_daily_data.parquet"
    
    if consolidated_path.exists():
        logger.info(f"  Loading consolidated daily data from {consolidated_path}")
        all_daily_df = pd.read_parquet(consolidated_path)
        
        # Ensure date column is datetime
        if 'date' in all_daily_df.columns:
            all_daily_df['date'] = pd.to_datetime(all_daily_df['date'])
            all_daily_df = all_daily_df.set_index('date')
        
        # Extract SPY data
        spy_ticker = settings.SPY_TICKER
        spy_df = all_daily_df[all_daily_df['ticker'] == spy_ticker].copy()
        spy_df = spy_df.sort_index()
        
        logger.info(f"  ✓ Loaded consolidated data: {all_daily_df['ticker'].nunique()} tickers")
        logger.info(f"  ✓ Loaded SPY data: {len(spy_df):,} days")
        
        use_consolidated = True
    else:
        # Fall back to individual files
        logger.info(f"  Consolidated data not found, using individual ticker files")
        spy_ticker = settings.SPY_TICKER
        spy_path = daily_data_dir / f"{spy_ticker}_daily.parquet"
        
        if not spy_path.exists():
            raise FileNotFoundError(
                f"SPY daily data not found at {spy_path}. "
                f"Please run Step 05 first."
            )
        
        spy_df = pd.read_parquet(spy_path)
        
        # Ensure SPY index is datetime and sorted
        if not isinstance(spy_df.index, pd.DatetimeIndex):
            spy_df.index = pd.to_datetime(spy_df.index)
        spy_df = spy_df.sort_index()
        
        logger.info(f"  ✓ Loaded SPY data: {len(spy_df):,} days")
        
        use_consolidated = False
        all_daily_df = None
    
    # Create a copy of the input DataFrame
    df = earnings_with_windows.copy()
    
    # Initialize new columns with NaN
    df['stock_return_30d'] = np.nan
    df['spy_return_30d'] = np.nan
    df['excess_return_30d'] = np.nan
    df['label_outperform_30d'] = pd.NA
    
    # Get column names
    ticker_col = settings.EARNINGS_TICKER_COLUMN
    
    # Check if required columns exist
    if ticker_col not in df.columns:
        logger.warning(f"Ticker column '{ticker_col}' not found")
        logger.info("Returning DataFrame with NaN targets")
        return df
    
    if 'has_full_post_window' not in df.columns:
        logger.warning("Column 'has_full_post_window' not found")
        logger.info("Returning DataFrame with NaN targets")
        return df
    
    # Track statistics
    total_events = len(df)
    events_with_coverage = df['has_full_post_window'].sum()
    successful_computations = 0
    failed_computations = 0
    missing_files = set()
    
    logger.info(f"\n  Processing {total_events:,} events...")
    logger.info(f"  Events with full post-window coverage: {events_with_coverage:,}")
    
    # Process each event
    for idx, row in df.iterrows():
        # Skip if no post-window coverage
        if not row.get('has_full_post_window', False):
            continue
        
        # Get ticker and window dates
        ticker = row[ticker_col]
        start_date = row['post_window_start']
        end_date = row['post_window_end']
        
        # Load ticker's daily data
        if use_consolidated:
            # Extract from consolidated data
            ticker_df = all_daily_df[all_daily_df['ticker'] == ticker].copy()
            
            if len(ticker_df) == 0:
                if ticker not in missing_files:
                    missing_files.add(ticker)
                    logger.warning(f"  ⚠ Daily data not found for {ticker}")
                failed_computations += 1
                continue
            
            ticker_df = ticker_df.sort_index()
        else:
            # Load from individual file
            ticker_path = daily_data_dir / f"{ticker}_daily.parquet"
            
            if not ticker_path.exists():
                if ticker not in missing_files:
                    missing_files.add(ticker)
                    logger.warning(f"  ⚠ Daily data not found for {ticker}")
                failed_computations += 1
                continue
            
            ticker_df = pd.read_parquet(ticker_path)
            
            # Ensure index is datetime and sorted
            if not isinstance(ticker_df.index, pd.DatetimeIndex):
                ticker_df.index = pd.to_datetime(ticker_df.index)
            ticker_df = ticker_df.sort_index()
        
        try:
            
            # Compute returns
            stock_return = compute_window_return(ticker_df, start_date, end_date)
            spy_return = compute_window_return(spy_df, start_date, end_date)
            
            # Check if both returns are valid
            if stock_return is None or spy_return is None:
                failed_computations += 1
                continue
            
            # Compute excess return
            excess_return = stock_return - spy_return
            
            # Compute label (1 if outperform, 0 otherwise)
            label = 1 if excess_return > 0 else 0
            
            # Store results
            df.loc[idx, 'stock_return_30d'] = stock_return
            df.loc[idx, 'spy_return_30d'] = spy_return
            df.loc[idx, 'excess_return_30d'] = excess_return
            df.loc[idx, 'label_outperform_30d'] = label
            
            successful_computations += 1
            
        except Exception as e:
            logger.warning(f"  ⚠ Error processing {ticker}: {str(e)}")
            failed_computations += 1
            continue
    
    # Log summary statistics
    logger.info(f"\n  Target Computation Summary:")
    logger.info(f"  Total events: {total_events:,}")
    logger.info(f"  Events with post-window coverage: {events_with_coverage:,}")
    logger.info(f"  Successful computations: {successful_computations:,}")
    logger.info(f"  Failed computations: {failed_computations:,}")
    
    if missing_files:
        logger.info(f"  Tickers with missing daily data: {len(missing_files)}")
    
    # Compute label distribution
    valid_labels = df['label_outperform_30d'].notna()
    if valid_labels.sum() > 0:
        outperform_count = (df.loc[valid_labels, 'label_outperform_30d'] == 1).sum()
        underperform_count = (df.loc[valid_labels, 'label_outperform_30d'] == 0).sum()
        
        logger.info(f"\n  Label Distribution:")
        logger.info(f"  Outperform (label=1): {outperform_count:,} ({100*outperform_count/valid_labels.sum():.1f}%)")
        logger.info(f"  Underperform (label=0): {underperform_count:,} ({100*underperform_count/valid_labels.sum():.1f}%)")
    
    # Return statistics
    valid_returns = df['excess_return_30d'].notna()
    if valid_returns.sum() > 0:
        mean_excess = df.loc[valid_returns, 'excess_return_30d'].mean()
        median_excess = df.loc[valid_returns, 'excess_return_30d'].median()
        std_excess = df.loc[valid_returns, 'excess_return_30d'].std()
        
        logger.info(f"\n  Excess Return Statistics:")
        logger.info(f"  Mean: {mean_excess:.4f} ({100*mean_excess:.2f}%)")
        logger.info(f"  Median: {median_excess:.4f} ({100*median_excess:.2f}%)")
        logger.info(f"  Std Dev: {std_excess:.4f} ({100*std_excess:.2f}%)")
    
    logger.info(f"\n  ✓ Target computation complete")
    
    return df


def create_binary_outperformance_label(
    excess_returns: pd.Series,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Create a binary classification label indicating whether the stock
    outperforms the benchmark based on excess return.

    Parameters
    ----------
    excess_returns : pd.Series
        Series of 30-day post-earnings excess returns (stock - SPY).
    threshold : float, optional
        Threshold above which an observation is considered an outperformance.
        By default 0.0, meaning strictly positive excess return.

    Returns
    -------
    pd.Series
        Binary label series where:
        1 indicates excess return > threshold (outperformance),
        0 indicates excess return <= threshold (underperformance or equal).
        
    Examples
    --------
    >>> excess_ret = pd.Series([0.05, -0.02, 0.0, 0.10])
    >>> labels = create_binary_outperformance_label(excess_ret)
    >>> labels.tolist()
    [1, 0, 0, 1]
    """
    label = (excess_returns > threshold).astype(int)
    label.name = "label_outperform"
    return label
