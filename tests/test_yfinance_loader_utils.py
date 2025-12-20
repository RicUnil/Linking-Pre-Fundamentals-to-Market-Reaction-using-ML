"""
Tests for yfinance loader utilities.

This module contains tests for the yfinance data loading utilities,
focusing on non-network operations like cache path building.
"""

import pytest
from pathlib import Path

import pandas as pd

from src.config import Settings
from src.data.yfinance_loader import build_cache_path, _standardize_columns


def test_build_cache_path_uses_ticker_and_dates(tmp_path, monkeypatch):
    """
    Test that build_cache_path creates a valid path with ticker and dates.
    
    This test verifies that:
    - The cache directory is created
    - The filename contains the ticker symbol
    - The filename contains the date range in YYYYMMDD format
    """
    settings = Settings()
    # Redirect cache dir for the test
    monkeypatch.setattr(settings, "YFINANCE_CACHE_DIR", tmp_path / "cache")
    
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-02-01")
    
    cache_path = build_cache_path("AAPL", start, end, settings)
    
    # Check that parent directory was created
    assert cache_path.parent.exists()
    
    # Check that filename contains expected components
    assert "AAPL" in cache_path.name
    assert "20200101" in cache_path.name
    assert "20200201" in cache_path.name
    assert cache_path.suffix == ".parquet"


def test_build_cache_path_handles_special_characters(tmp_path, monkeypatch):
    """
    Test that build_cache_path handles special characters in ticker symbols.
    
    Some tickers contain special characters like ^ or / which need to be
    replaced with safe characters for filenames.
    """
    settings = Settings()
    monkeypatch.setattr(settings, "YFINANCE_CACHE_DIR", tmp_path / "cache")
    
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-12-31")
    
    # Test ticker with special character
    cache_path = build_cache_path("^GSPC", start, end, settings)
    
    # Check that special character was replaced
    assert "^" not in cache_path.name
    assert "_GSPC" in cache_path.name


def test_standardize_columns():
    """
    Test that _standardize_columns converts column names to snake_case.
    
    This test verifies that yfinance column names are properly standardized.
    """
    # Create a sample DataFrame with typical yfinance column names
    df = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [105, 106, 107],
        'Low': [99, 100, 101],
        'Close': [103, 104, 105],
        'Adj Close': [102, 103, 104],
        'Volume': [1000, 1100, 1200]
    })
    
    # Standardize columns
    result = _standardize_columns(df)
    
    # Check that all columns are lowercase
    assert all(col.islower() for col in result.columns)
    
    # Check specific transformations
    assert 'open' in result.columns
    assert 'high' in result.columns
    assert 'low' in result.columns
    assert 'close' in result.columns
    assert 'adj_close' in result.columns
    assert 'volume' in result.columns
    
    # Check that data is preserved
    assert len(result) == len(df)
    assert result['open'].tolist() == [100, 101, 102]


def test_standardize_columns_preserves_data():
    """
    Test that _standardize_columns doesn't modify the original DataFrame.
    """
    df = pd.DataFrame({
        'Open': [100, 101],
        'Close': [103, 104]
    })
    
    original_columns = df.columns.tolist()
    
    # Standardize columns
    result = _standardize_columns(df)
    
    # Check that original DataFrame is unchanged
    assert df.columns.tolist() == original_columns
    
    # Check that result has different columns
    assert result.columns.tolist() != original_columns
