"""
Data loading utilities for Capital IQ earnings data.

This module provides functions to load the main earnings dataset and quarterly
supplementary files. It handles CSV parsing, date column detection, and basic
validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from src.config import Settings


logger = logging.getLogger(__name__)


def _detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that likely contain dates based on naming patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.
        
    Returns
    -------
    List[str]
        List of column names that appear to contain dates.
    """
    date_keywords = ['DATE', 'ANNOUNCEMENT', 'EARNINGS', 'FILING', 'QUARTER']
    date_columns = []
    
    for col in df.columns:
        col_upper = str(col).upper()
        if any(keyword in col_upper for keyword in date_keywords):
            # Check if it ends with DATE or contains DATE
            if 'DATE' in col_upper:
                date_columns.append(col)
    
    return date_columns


def _parse_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """
    Attempt to parse specified columns as dates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential date columns.
    date_columns : List[str]
        List of column names to parse as dates.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with parsed date columns (where successful).
    """
    df_copy = df.copy()
    
    for col in date_columns:
        try:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            logger.info(f"  ✓ Parsed date column: {col}")
        except Exception as e:
            logger.warning(f"  ⚠ Could not parse {col} as date: {e}")
    
    return df_copy


def load_raw_earnings_data(settings: Settings) -> pd.DataFrame:
    """
    Load the main earnings dataset from RAW_DATA.csv.
    
    This function loads the primary earnings events dataset, detects and parses
    date columns, and performs basic validation.
    
    Parameters
    ----------
    settings : Settings
        Global project settings containing the data directory path.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw earnings events.
        
    Raises
    ------
    FileNotFoundError
        If RAW_DATA.csv does not exist in the data directory.
    """
    file_path = settings.DATA_DIR / "RAW_DATA.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"RAW_DATA.csv not found at {file_path}. "
            f"Please ensure the file exists in {settings.DATA_DIR}."
        )
    
    logger.info(f"Loading RAW_DATA.csv from {file_path}")
    
    # Load CSV with low_memory=False to avoid dtype warnings
    # Use semicolon as delimiter (Capital IQ format)
    df = pd.read_csv(file_path, sep=';', low_memory=False)
    
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Columns (first 10): {list(df.columns[:10])}")
    
    # Detect and parse date columns
    date_columns = _detect_date_columns(df)
    if date_columns:
        logger.info(f"  Detected potential date columns: {date_columns}")
        df = _parse_date_columns(df, date_columns)
    
    return df


def load_quarter_file(settings: Settings, quarter: int) -> pd.DataFrame:
    """
    Load a quarterly dataset (Quarter_1.csv, Quarter_2.csv, etc.).
    
    Parameters
    ----------
    settings : Settings
        Global project settings.
    quarter : int
        Quarter number in {1, 2, 3, 4}.
        
    Returns
    -------
    pd.DataFrame
        DataFrame for the requested quarter.
        
    Raises
    ------
    ValueError
        If quarter is not in {1, 2, 3, 4}.
    FileNotFoundError
        If the quarterly file does not exist.
    """
    if quarter not in {1, 2, 3, 4}:
        raise ValueError(f"Quarter must be in {{1, 2, 3, 4}}, got {quarter}")
    
    file_path = settings.DATA_DIR / f"Quarter_{quarter}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Quarter_{quarter}.csv not found at {file_path}. "
            f"Please ensure the file exists in {settings.DATA_DIR}."
        )
    
    logger.info(f"Loading Quarter_{quarter}.csv from {file_path}")
    
    # Load CSV with semicolon delimiter (Capital IQ format)
    df = pd.read_csv(file_path, sep=';', low_memory=False)
    
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Detect and parse date columns
    date_columns = _detect_date_columns(df)
    if date_columns:
        logger.info(f"  Detected potential date columns: {date_columns}")
        df = _parse_date_columns(df, date_columns)
    
    return df


def load_all_quarter_files(settings: Settings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all four quarter CSV files.
    
    Parameters
    ----------
    settings : Settings
        Global project settings.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames for quarters 1–4, in order.
        
    Raises
    ------
    FileNotFoundError
        If any quarterly file is missing.
    """
    logger.info("Loading all quarterly files...")
    
    q1 = load_quarter_file(settings, 1)
    q2 = load_quarter_file(settings, 2)
    q3 = load_quarter_file(settings, 3)
    q4 = load_quarter_file(settings, 4)
    
    logger.info("✓ All quarterly files loaded successfully")
    
    return q1, q2, q3, q4


def load_benchmark_data(settings: Settings) -> pd.DataFrame:
    """
    Load the benchmark (SPY) dataset from BENCHMARK.csv.
    
    This function loads the benchmark time series data, typically containing
    SPY total return index or similar market benchmark data.
    
    Parameters
    ----------
    settings : Settings
        Global project settings containing the data directory path.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the benchmark time series.
        
    Raises
    ------
    FileNotFoundError
        If BENCHMARK.csv does not exist in the data directory.
        
    Notes
    -----
    The function attempts to:
    - Read data/BENCHMARK.csv
    - Parse any date-like column (e.g. 'DATE', 'TRADING_DATE')
    - Log the shape and column names
    """
    file_path = settings.DATA_DIR / "BENCHMARK.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"BENCHMARK.csv not found at {file_path}. "
            f"Please ensure the file exists in {settings.DATA_DIR}."
        )
    
    logger.info(f"Loading BENCHMARK.csv from {file_path}")
    
    # Load CSV with semicolon delimiter (Capital IQ format)
    df = pd.read_csv(file_path, sep=';', low_memory=False)
    
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Detect and parse date columns
    date_columns = _detect_date_columns(df)
    if date_columns:
        logger.info(f"  Detected potential date columns: {date_columns}")
        df = _parse_date_columns(df, date_columns)
    
    return df
