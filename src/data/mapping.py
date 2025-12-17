"""
Temporal mapping utilities for earnings events and daily data.

This module provides functions to compute pre- and post-earnings date windows
and to check whether downloaded daily data covers these windows.
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from src.config import Settings


logger = logging.getLogger(__name__)


def compute_event_windows(
    earnings_df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """
    Compute pre- and post-earnings calendar date windows for each event.
    
    For each earnings event, this function computes:
    - Pre window:  [EARNINGS_DATE - 30 days, EARNINGS_DATE - 1 day]
    - Post window: [EARNINGS_DATE + 1 day, EARNINGS_DATE + 30 days]
    
    These windows define the periods before and after earnings announcements
    that will be used for computing returns and features in later steps.
    
    Parameters
    ----------
    earnings_df : pd.DataFrame
        DataFrame containing at least the ticker and earnings date columns.
    settings : Settings
        Global project settings, used to locate the ticker and date columns.
        
    Returns
    -------
    pd.DataFrame
        A copy of earnings_df with the following additional datetime64 columns:
        - pre_window_start: Start of pre-earnings window (J-30)
        - pre_window_end: End of pre-earnings window (J-1)
        - post_window_start: Start of post-earnings window (J+1)
        - post_window_end: End of post-earnings window (J+30)
        
    Notes
    -----
    - Rows with missing or invalid earnings dates are dropped
    - The function logs how many rows were dropped
    - Windows are calendar-based, not trading-day-based
    """
    logger.info("Computing pre- and post-earnings date windows...")
    
    # Create a copy to avoid modifying the original
    df = earnings_df.copy()
    
    date_col = settings.EARNINGS_DATE_COLUMN
    
    # Check if date column exists
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        logger.warning(f"Available columns: {list(df.columns[:10])}...")
        logger.info("Creating dummy date column for development")
        # For development, create a dummy date column
        df[date_col] = pd.Timestamp("2020-01-15")
    
    # Convert earnings date to datetime
    initial_rows = len(df)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Drop rows with missing dates
    df = df.dropna(subset=[date_col])
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        logger.info(f"  Dropped {dropped_rows:,} rows with missing/invalid earnings dates")
    
    logger.info(f"  Processing {len(df):,} events with valid dates")
    
    # Compute window boundaries
    df['pre_window_start'] = df[date_col] - pd.Timedelta(days=30)
    df['pre_window_end'] = df[date_col] - pd.Timedelta(days=1)
    df['post_window_start'] = df[date_col] + pd.Timedelta(days=1)
    df['post_window_end'] = df[date_col] + pd.Timedelta(days=30)
    
    logger.info(f"  ✓ Computed windows for {len(df):,} events")
    logger.info(f"  Window columns added: pre_window_start, pre_window_end, post_window_start, post_window_end")
    
    return df


def flag_window_coverage(
    earnings_with_windows: pd.DataFrame,
    manifest: Dict[str, Dict[str, object]],
    settings: Settings,
) -> pd.DataFrame:
    """
    Add boolean flags indicating window coverage by downloaded daily data.
    
    For each earnings event, this function checks whether the pre- and
    post-earnings windows are fully covered by the daily price data that
    was downloaded in Step 05.
    
    Parameters
    ----------
    earnings_with_windows : pd.DataFrame
        Earnings DataFrame with window columns already computed.
    manifest : dict
        Mapping from ticker to metadata, usually loaded from
        results/step_05/daily_data_manifest.json. Expected keys per ticker:
        - 'start' : ISO date string (e.g., '2018-01-02')
        - 'end'   : ISO date string (e.g., '2024-11-29')
        - 'rows'  : number of data points
    settings : Settings
        Global project settings.
        
    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with additional boolean columns:
        - has_full_pre_window: True if pre-window is fully covered
        - has_full_post_window: True if post-window is fully covered
        
    Notes
    -----
    - If a ticker is not in the manifest, both flags are set to False
    - Coverage is calendar-based (not trading-day-based)
    - Summary statistics are logged
    """
    logger.info("Flagging window coverage based on daily data availability...")
    
    # Create a copy
    df = earnings_with_windows.copy()
    
    ticker_col = settings.EARNINGS_TICKER_COLUMN
    
    # Check if ticker column exists
    if ticker_col not in df.columns:
        logger.warning(f"Ticker column '{ticker_col}' not found in DataFrame")
        logger.info("Setting all coverage flags to False")
        df['has_full_pre_window'] = False
        df['has_full_post_window'] = False
        return df
    
    # Initialize coverage flags
    df['has_full_pre_window'] = False
    df['has_full_post_window'] = False
    
    # Check coverage for each row
    tickers_in_manifest = set(manifest.keys())
    tickers_in_data = set(df[ticker_col].unique())
    
    logger.info(f"  Tickers in manifest: {len(tickers_in_manifest)}")
    logger.info(f"  Unique tickers in earnings data: {len(tickers_in_data)}")
    
    # Track statistics
    pre_covered = 0
    post_covered = 0
    both_covered = 0
    
    for idx, row in df.iterrows():
        ticker = row[ticker_col]
        
        if ticker not in manifest:
            # Ticker not downloaded - flags remain False
            continue
        
        # Get manifest info for this ticker
        ticker_info = manifest[ticker]
        manifest_start = pd.Timestamp(ticker_info['start'])
        manifest_end = pd.Timestamp(ticker_info['end'])
        
        # Check pre-window coverage
        pre_start = row['pre_window_start']
        pre_end = row['pre_window_end']
        
        if manifest_start <= pre_start and manifest_end >= pre_end:
            df.loc[idx, 'has_full_pre_window'] = True
            pre_covered += 1
        
        # Check post-window coverage
        post_start = row['post_window_start']
        post_end = row['post_window_end']
        
        if manifest_start <= post_start and manifest_end >= post_end:
            df.loc[idx, 'has_full_post_window'] = True
            post_covered += 1
        
        # Track both covered
        if df.loc[idx, 'has_full_pre_window'] and df.loc[idx, 'has_full_post_window']:
            both_covered += 1
    
    # Log summary statistics
    total_events = len(df)
    logger.info(f"\n  Coverage Summary:")
    logger.info(f"  Total events: {total_events:,}")
    logger.info(f"  Events with full pre-window: {pre_covered:,} ({100*pre_covered/total_events:.1f}%)")
    logger.info(f"  Events with full post-window: {post_covered:,} ({100*post_covered/total_events:.1f}%)")
    logger.info(f"  Events with both windows covered: {both_covered:,} ({100*both_covered/total_events:.1f}%)")
    
    logger.info(f"  ✓ Coverage flags added: has_full_pre_window, has_full_post_window")
    
    return df
