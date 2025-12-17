"""
Load consolidated daily data efficiently.

This module provides utilities to load the consolidated daily price data
that was created by the consolidation script.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import Settings


logger = logging.getLogger(__name__)


def load_all_daily_data(settings: Settings) -> pd.DataFrame:
    """
    Load all consolidated daily data.
    
    Parameters
    ----------
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all daily data for all tickers
        Columns: ticker, date (index), open, high, low, close, adj_close, volume
    """
    cache_path = settings.get_step_results_dir(5) / "cache" / "all_daily_data.parquet"
    
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Consolidated data not found at {cache_path}. "
            f"Please run: python consolidate_project.py"
        )
    
    logger.info(f"Loading consolidated daily data from {cache_path}")
    df = pd.read_parquet(cache_path)
    logger.info(f"  Loaded {len(df):,} rows for {df['ticker'].nunique()} tickers")
    
    return df


def load_daily_data_as_dict(settings: Settings) -> Dict[str, pd.DataFrame]:
    """
    Load consolidated daily data as a dictionary of DataFrames.
    
    This is compatible with the original format used by the pipeline.
    
    Parameters
    ----------
    settings : Settings
        Global project settings
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping ticker symbol to its daily DataFrame
        Each DataFrame has columns: open, high, low, close, adj_close, volume
    """
    df = load_all_daily_data(settings)
    
    # Group by ticker and create dictionary
    data_dict = {}
    for ticker, group in df.groupby('ticker'):
        # Drop ticker column and keep only price data
        ticker_df = group.drop('ticker', axis=1)
        data_dict[ticker] = ticker_df
    
    logger.info(f"  Created dictionary with {len(data_dict)} tickers")
    
    return data_dict


def load_ticker_data(ticker: str, settings: Settings) -> Optional[pd.DataFrame]:
    """
    Load daily data for a specific ticker.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame or None
        Daily data for the ticker, or None if not found
    """
    df = load_all_daily_data(settings)
    
    ticker_data = df[df['ticker'] == ticker]
    
    if ticker_data.empty:
        logger.warning(f"No data found for ticker: {ticker}")
        return None
    
    # Drop ticker column
    ticker_data = ticker_data.drop('ticker', axis=1)
    
    return ticker_data


def get_available_tickers(settings: Settings) -> list:
    """
    Get list of all available tickers in consolidated data.
    
    Parameters
    ----------
    settings : Settings
        Global project settings
        
    Returns
    -------
    list
        List of ticker symbols
    """
    df = load_all_daily_data(settings)
    return sorted(df['ticker'].unique().tolist())


# Backward compatibility: provide the same interface as before
def load_daily_data_from_step_05(settings: Settings) -> Dict[str, pd.DataFrame]:
    """
    Load daily data with backward compatibility.
    
    This function tries to load from the consolidated file first,
    and falls back to individual files if needed.
    
    Parameters
    ----------
    settings : Settings
        Global project settings
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping ticker to daily DataFrame
    """
    cache_path = settings.get_step_results_dir(5) / "cache" / "all_daily_data.parquet"
    
    if cache_path.exists():
        # Load from consolidated file
        logger.info("Loading from consolidated data file...")
        return load_daily_data_as_dict(settings)
    else:
        # Fall back to individual files
        logger.info("Loading from individual ticker files...")
        step_05_dir = settings.get_step_results_dir(5)
        
        # Check archive directory
        archive_dir = step_05_dir / "archive"
        if archive_dir.exists():
            ticker_files = list(archive_dir.glob("*_daily.parquet"))
        else:
            ticker_files = list(step_05_dir.glob("*_daily.parquet"))
        
        data_dict = {}
        for ticker_file in ticker_files:
            ticker = ticker_file.stem.replace("_daily", "")
            try:
                df = pd.read_parquet(ticker_file)
                data_dict[ticker] = df
            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")
        
        logger.info(f"  Loaded {len(data_dict)} tickers from individual files")
        return data_dict
