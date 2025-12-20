"""
Pre-earnings data fetching utilities.

This module orchestrates downloading daily price/volume data for tickers
appearing in the earnings dataset, based on their earnings announcement dates.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd

from src.config import Settings
from src.data.yfinance_loader import download_price_volume


logger = logging.getLogger(__name__)


def compute_global_date_range_for_ticker(
    earnings_df: pd.DataFrame,
    ticker: str,
    settings: Settings,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Compute the start/end dates for yfinance downloads for a given ticker.
    
    The date range is computed based on all earnings announcements for this
    ticker in the dataset:
    - Start = min(EARNINGS_DATE) - 180 days
    - End   = max(EARNINGS_DATE) + 30 days
    
    This ensures we have sufficient historical data before the first earnings
    event and post-earnings data after the last event.
    
    Parameters
    ----------
    earnings_df : pd.DataFrame
        Main earnings dataset containing ticker and date columns.
    ticker : str
        Ticker symbol to compute range for.
    settings : Settings
        Global project settings.
        
    Returns
    -------
    Tuple[pd.Timestamp, pd.Timestamp]
        (start_date, end_date) for downloading daily data.
        
    Raises
    ------
    ValueError
        If no earnings dates found for the ticker.
    """
    ticker_col = settings.EARNINGS_TICKER_COLUMN
    date_col = settings.EARNINGS_DATE_COLUMN
    
    # Filter to this ticker
    ticker_data = earnings_df[earnings_df[ticker_col] == ticker]
    
    if len(ticker_data) == 0:
        raise ValueError(f"No earnings data found for ticker {ticker}")
    
    # Get date range
    dates = ticker_data[date_col]
    min_date = dates.min()
    max_date = dates.max()
    
    # Add buffers
    start = min_date - pd.Timedelta(days=180)
    end = max_date + pd.Timedelta(days=30)
    
    return start, end


def fetch_daily_data_for_tickers(
    earnings_df: pd.DataFrame,
    settings: Settings,
) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data for tickers in the earnings dataset plus SPY.
    
    This function:
    1. Identifies unique tickers in the earnings dataset
    2. Limits to MAX_TICKERS_DAILY_DOWNLOAD for development
    3. Computes appropriate date ranges for each ticker
    4. Downloads daily data using yfinance with caching
    5. Also downloads SPY benchmark data
    
    Parameters
    ----------
    earnings_df : pd.DataFrame
        Main earnings dataset from Step 03.
    settings : Settings
        Global project settings.
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping from ticker symbol to its daily price/volume DataFrame.
        Each DataFrame is indexed by date with columns:
        open, high, low, close, adj_close, volume.
        
    Notes
    -----
    - Rows with missing ticker or date are dropped
    - Only the first MAX_TICKERS_DAILY_DOWNLOAD tickers are processed
    - SPY is always included in the results
    - If the expected columns don't exist, uses a demo list of tickers
    """
    ticker_col = settings.EARNINGS_TICKER_COLUMN
    date_col = settings.EARNINGS_DATE_COLUMN
    
    logger.info("=" * 70)
    logger.info("Fetching daily price/volume data for earnings tickers")
    logger.info("=" * 70)
    
    # Check if expected columns exist
    has_ticker_col = ticker_col in earnings_df.columns
    has_date_col = date_col in earnings_df.columns
    
    if not has_ticker_col or not has_date_col:
        logger.warning(f"Expected columns not found in earnings data")
        logger.warning(f"  Looking for: {ticker_col}, {date_col}")
        logger.warning(f"  Available columns: {list(earnings_df.columns[:10])}...")
        logger.info(f"\nUsing demo ticker list for development")
        
        # Use a hardcoded list of well-known tickers for demo
        demo_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "TSLA", "NVDA", "JPM", "V", "WMT",
            "JNJ", "PG", "MA", "HD", "DIS",
            "BAC", "ADBE", "CRM", "NFLX", "INTC"
        ]
        
        selected_tickers = demo_tickers[:settings.MAX_TICKERS_DAILY_DOWNLOAD]
        logger.info(f"Demo tickers: {selected_tickers}")
        
        # Use a fixed date range for demo
        global_start = pd.Timestamp("2018-01-01")
        global_end = pd.Timestamp("2024-12-01")
        
    else:
        # Original logic when columns exist
        # Clean the earnings data
        initial_rows = len(earnings_df)
        clean_df = earnings_df.dropna(subset=[ticker_col, date_col]).copy()
        dropped_rows = initial_rows - len(clean_df)
        
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows:,} rows with missing ticker or date")
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(clean_df[date_col]):
            clean_df[date_col] = pd.to_datetime(clean_df[date_col], errors='coerce')
            # Drop any rows where date parsing failed
            clean_df = clean_df.dropna(subset=[date_col])
        
        # Get unique tickers
        unique_tickers = clean_df[ticker_col].unique()
        logger.info(f"Found {len(unique_tickers):,} unique tickers in earnings dataset")
        
        # Limit number of tickers for development
        max_tickers = settings.MAX_TICKERS_DAILY_DOWNLOAD
        if len(unique_tickers) > max_tickers:
            logger.info(f"Limiting to first {max_tickers} tickers for development")
            selected_tickers = unique_tickers[:max_tickers]
            ignored_count = len(unique_tickers) - max_tickers
            logger.info(f"  (Ignoring {ignored_count:,} additional tickers)")
        else:
            selected_tickers = unique_tickers
        
        logger.info(f"\nSelected tickers: {list(selected_tickers)}")
        
        # Compute global date range
        global_start = clean_df[date_col].min() - pd.Timedelta(days=180)
        global_end = clean_df[date_col].max() + pd.Timedelta(days=30)
    
    # Dictionary to store results
    daily_data: Dict[str, pd.DataFrame] = {}
    
    # Download data for each ticker
    logger.info(f"\nDownloading daily data for {len(selected_tickers)} tickers...")
    logger.info(f"Date range: {global_start.date()} to {global_end.date()}")
    logger.info("-" * 70)
    
    for i, ticker in enumerate(selected_tickers, 1):
        logger.info(f"[{i}/{len(selected_tickers)}] Processing {ticker}")
        
        try:
            # Download data
            df = download_price_volume(ticker, global_start, global_end, settings)
            
            if not df.empty:
                daily_data[ticker] = df
                logger.info(f"  ✓ Downloaded {len(df):,} days of data")
            else:
                logger.warning(f"  ⚠ No data available for {ticker}")
                
        except Exception as e:
            logger.error(f"  ✗ Error processing {ticker}: {str(e)}")
            continue
    
    # Download SPY benchmark data
    logger.info(f"\n[SPY] Processing benchmark")
    logger.info(f"  Date range: {global_start.date()} to {global_end.date()}")
    
    spy_ticker = settings.SPY_TICKER
    spy_df = download_price_volume(spy_ticker, global_start, global_end, settings)
    
    if not spy_df.empty:
        daily_data[spy_ticker] = spy_df
        logger.info(f"  ✓ Downloaded {len(spy_df):,} days of SPY data")
    else:
        logger.warning(f"  ⚠ No SPY data available")
    
    logger.info("-" * 70)
    logger.info(f"✓ Successfully downloaded data for {len(daily_data)} tickers")
    
    return daily_data
