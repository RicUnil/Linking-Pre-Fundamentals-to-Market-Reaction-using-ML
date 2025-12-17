"""
yfinance data loader with local caching and rate limiting.

This module provides utilities for downloading daily OHLCV data from Yahoo Finance
with local Parquet caching to avoid redundant API calls.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import Settings


logger = logging.getLogger(__name__)


def build_cache_path(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: Settings,
) -> Path:
    """
    Build the cache path for a given ticker and date range.
    
    The cache filename includes the ticker symbol and date range to ensure
    uniqueness and easy identification.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'AAPL', 'SPY').
    start : pd.Timestamp
        Start date for the data range.
    end : pd.Timestamp
        End date for the data range.
    settings : Settings
        Global project settings containing cache directory path.
        
    Returns
    -------
    Path
        Full path to the cache file.
        
    Notes
    -----
    The cache directory is created if it doesn't exist.
    Filename format: {ticker}_{start_YYYYMMDD}_{end_YYYYMMDD}.parquet
    """
    # Ensure cache directory exists
    cache_dir = settings.YFINANCE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename with safe characters
    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')
    
    # Replace any special characters in ticker (e.g., ^GSPC -> _GSPC)
    safe_ticker = ticker.replace('^', '_').replace('/', '_')
    
    filename = f"{safe_ticker}_{start_str}_{end_str}.parquet"
    return cache_dir / filename


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common yfinance column names to snake_case.
    
    This function converts column names like 'Adj Close', 'Open', 'High', etc.
    to lowercase snake_case format for consistency.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with yfinance column names.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
        
    Examples
    --------
    'Adj Close' -> 'adj_close'
    'Open' -> 'open'
    'High' -> 'high'
    'Volume' -> 'volume'
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Handle MultiIndex columns (can happen with yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex to single level (use first level)
        df.columns = df.columns.get_level_values(0)
    
    # Convert to lowercase and replace spaces with underscores
    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
    
    return df


def download_price_volume(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: Settings,
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a given ticker and date range using yfinance.
    
    This function implements local caching to avoid redundant API calls and
    includes basic rate limiting to be respectful to the Yahoo Finance API.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol to download (e.g., 'AAPL', 'SPY').
    start : pd.Timestamp
        Start date (inclusive).
    end : pd.Timestamp
        End date (inclusive).
    settings : Settings
        Global project settings.
    sleep_seconds : float, optional
        Time to sleep after a fresh download to avoid hammering the API.
        Default is 0.25 seconds.
        
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with standardized column names
        (open, high, low, close, adj_close, volume).
        Returns empty DataFrame if download fails.
        
    Notes
    -----
    - If cached data exists, it is loaded from disk without hitting the API.
    - Fresh downloads are saved to cache as Parquet files.
    - Column names are standardized to snake_case.
    - The index is ensured to be datetime type.
    """
    # Build cache path
    cache_path = build_cache_path(ticker, start, end, settings)
    
    # Check if cache exists
    if cache_path.exists():
        logger.info(f"  Loading {ticker} from cache: {cache_path.name}")
        df = pd.read_parquet(cache_path)
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return df
    
    # Cache miss - download from yfinance
    logger.info(f"  Downloading {ticker} from yfinance ({start.date()} to {end.date()})")
    
    try:
        # Download data
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            interval="1d"
        )
        
        # Check if download was successful
        if df.empty:
            logger.warning(f"  ⚠ No data returned for {ticker} in range {start.date()} to {end.date()}")
            return pd.DataFrame()
        
        # Standardize column names
        df = _standardize_columns(df)
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Save to cache
        df.to_parquet(cache_path, engine='pyarrow')
        logger.info(f"  ✓ Cached {ticker} to {cache_path.name} ({len(df)} rows)")
        
        # Rate limiting
        time.sleep(sleep_seconds)
        
        return df
        
    except Exception as e:
        logger.error(f"  ✗ Error downloading {ticker}: {str(e)}")
        return pd.DataFrame()
