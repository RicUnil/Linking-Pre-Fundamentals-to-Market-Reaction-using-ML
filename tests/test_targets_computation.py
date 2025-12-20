"""
Tests for target computation (30-day post-earnings returns).

This module contains unit tests for the return computation and target
construction functions in Step 07.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import Settings
from src.preprocessing.targets import (
    _select_price_column,
    compute_window_return,
    add_post_earnings_targets
)


def test_select_price_column_prefers_adj_close():
    """
    Test that _select_price_column prefers adj_close over close.
    """
    df = pd.DataFrame({
        'adj_close': [100, 101],
        'close': [100, 101],
        'open': [99, 100]
    })
    
    result = _select_price_column(df)
    assert result == 'adj_close'


def test_select_price_column_falls_back_to_close():
    """
    Test that _select_price_column uses close if adj_close is not available.
    """
    df = pd.DataFrame({
        'close': [100, 101],
        'open': [99, 100]
    })
    
    result = _select_price_column(df)
    assert result == 'close'


def test_select_price_column_raises_on_missing_columns():
    """
    Test that _select_price_column raises ValueError if neither column exists.
    """
    df = pd.DataFrame({
        'open': [99, 100],
        'high': [101, 102]
    })
    
    with pytest.raises(ValueError, match="Neither 'adj_close' nor 'close' found"):
        _select_price_column(df)


def test_compute_window_return_simple_case():
    """
    Test compute_window_return with a simple case.
    
    100 -> 121 should give 21% return.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"adj_close": [100.0, 110.0, 121.0]},
        index=dates,
    )
    
    r = compute_window_return(df, dates[0], dates[-1])
    
    # 100 -> 121 => 21% return
    assert r is not None
    assert np.isclose(r, 0.21, atol=1e-6)


def test_compute_window_return_uses_first_and_last():
    """
    Test that compute_window_return uses first and last prices in window.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"adj_close": [100.0, 105.0, 95.0, 110.0, 120.0]},
        index=dates,
    )
    
    # Should use 100 (first) and 120 (last), ignoring middle values
    r = compute_window_return(df, dates[0], dates[-1])
    
    assert r is not None
    assert np.isclose(r, 0.20, atol=1e-6)  # 100 -> 120 = 20%


def test_compute_window_return_returns_none_for_insufficient_data():
    """
    Test that compute_window_return returns None if fewer than 2 data points.
    """
    dates = pd.date_range("2020-01-01", periods=1, freq="D")
    df = pd.DataFrame(
        {"adj_close": [100.0]},
        index=dates,
    )
    
    r = compute_window_return(df, dates[0], dates[0])
    
    assert r is None


def test_compute_window_return_handles_missing_prices():
    """
    Test that compute_window_return returns None if prices are NaN.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"adj_close": [np.nan, 110.0, 121.0]},
        index=dates,
    )
    
    r = compute_window_return(df, dates[0], dates[-1])
    
    assert r is None


def test_compute_window_return_handles_zero_prices():
    """
    Test that compute_window_return returns None if prices are zero or negative.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"adj_close": [0.0, 110.0, 121.0]},
        index=dates,
    )
    
    r = compute_window_return(df, dates[0], dates[-1])
    
    assert r is None


def test_compute_window_return_filters_to_window():
    """
    Test that compute_window_return only uses data within the specified window.
    """
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"adj_close": [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0]},
        index=dates,
    )
    
    # Use only days 2-5 (indices 2-5)
    r = compute_window_return(df, dates[2], dates[5])
    
    # Should use 110 (index 2) and 125 (index 5)
    expected = (125.0 / 110.0) - 1.0
    assert r is not None
    assert np.isclose(r, expected, atol=1e-6)


def test_add_post_earnings_targets_creates_columns():
    """
    Test that add_post_earnings_targets creates the expected columns.
    """
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "TICKER"
    settings.EARNINGS_DATE_COLUMN = "EARNINGS_DATE"
    
    # Simple synthetic earnings event
    earnings_df = pd.DataFrame({
        "TICKER": ["TEST"],
        "EARNINGS_DATE": [pd.Timestamp("2020-01-02")],
        "pre_window_start": [pd.Timestamp("2019-12-03")],
        "pre_window_end": [pd.Timestamp("2020-01-01")],
        "post_window_start": [pd.Timestamp("2020-01-03")],
        "post_window_end": [pd.Timestamp("2020-02-01")],
        "has_full_pre_window": [True],
        "has_full_post_window": [False],  # Set to False to avoid file I/O
    })
    
    result = add_post_earnings_targets(earnings_df, settings)
    
    # Check that new columns were created
    assert 'stock_return_30d' in result.columns
    assert 'spy_return_30d' in result.columns
    assert 'excess_return_30d' in result.columns
    assert 'label_outperform_30d' in result.columns
    
    # Since has_full_post_window is False, values should be NaN
    assert pd.isna(result.loc[0, 'stock_return_30d'])
    assert pd.isna(result.loc[0, 'excess_return_30d'])


def test_add_post_earnings_targets_preserves_original_data():
    """
    Test that add_post_earnings_targets doesn't modify original columns.
    """
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "TICKER"
    settings.EARNINGS_DATE_COLUMN = "EARNINGS_DATE"
    
    earnings_df = pd.DataFrame({
        "TICKER": ["TEST"],
        "EARNINGS_DATE": [pd.Timestamp("2020-01-02")],
        "SOME_VALUE": [42],
        "post_window_start": [pd.Timestamp("2020-01-03")],
        "post_window_end": [pd.Timestamp("2020-02-01")],
        "has_full_post_window": [False],
    })
    
    result = add_post_earnings_targets(earnings_df, settings)
    
    # Check that original data is preserved
    assert result.loc[0, "TICKER"] == "TEST"
    assert result.loc[0, "SOME_VALUE"] == 42
    assert result.loc[0, "EARNINGS_DATE"] == pd.Timestamp("2020-01-02")


def test_compute_window_return_negative_return():
    """
    Test that compute_window_return correctly handles negative returns.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"adj_close": [100.0, 95.0, 80.0]},
        index=dates,
    )
    
    r = compute_window_return(df, dates[0], dates[-1])
    
    # 100 -> 80 => -20% return
    assert r is not None
    assert np.isclose(r, -0.20, atol=1e-6)
