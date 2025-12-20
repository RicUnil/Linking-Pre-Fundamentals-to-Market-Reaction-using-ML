"""
Unit tests for market-based feature engineering.

This module tests the helper functions used to compute market features
such as momentum, volatility, and volume statistics.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features import (
    compute_momentum_window,
    compute_pre_earnings_stats,
    compute_daily_returns,
    _select_price_column,
)


def test_select_price_column_adj_close():
    """Test that adj_close is preferred when available."""
    df = pd.DataFrame({
        "adj_close": [100.0, 101.0],
        "close": [100.0, 101.0],
    })
    
    col = _select_price_column(df)
    assert col == "adj_close"


def test_select_price_column_close_only():
    """Test that close is used when adj_close is not available."""
    df = pd.DataFrame({
        "close": [100.0, 101.0],
    })
    
    col = _select_price_column(df)
    assert col == "close"


def test_select_price_column_missing():
    """Test that ValueError is raised when neither column is present."""
    df = pd.DataFrame({
        "open": [100.0, 101.0],
    })
    
    with pytest.raises(ValueError, match="Neither 'adj_close' nor 'close' found"):
        _select_price_column(df)


def test_compute_daily_returns_basic():
    """Test basic daily returns computation."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"adj_close": [100.0, 102.0, 101.0, 103.0, 104.0]},
        index=dates,
    )
    
    returns = compute_daily_returns(df)
    
    # First return should be NaN
    assert pd.isna(returns.iloc[0])
    
    # Second return: (102 - 100) / 100 = 0.02
    assert np.isclose(returns.iloc[1], 0.02, atol=1e-6)
    
    # Third return: (101 - 102) / 102 ≈ -0.0098
    assert np.isclose(returns.iloc[2], -0.0098, atol=1e-4)


def test_compute_daily_returns_empty():
    """Test that empty dataframe returns empty series."""
    df = pd.DataFrame()
    returns = compute_daily_returns(df)
    
    assert len(returns) == 0
    assert returns.dtype == "float64"


def test_compute_momentum_window_basic():
    """Test basic momentum computation."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"adj_close": [100.0, 105.0, 110.0, 115.0, 120.0]},
        index=dates,
    )

    m = compute_momentum_window(df, end_date=dates[-1], window_days=10)
    
    # 100 -> 120 => +20%
    assert m is not None
    assert np.isclose(m, 0.20, atol=1e-6)


def test_compute_momentum_window_30_days():
    """Test 30-day momentum computation."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    prices = np.linspace(100, 120, 40)  # Linear growth from 100 to 120
    df = pd.DataFrame(
        {"adj_close": prices},
        index=dates,
    )

    # Compute momentum ending at day 39 (last day), looking back 30 days
    m = compute_momentum_window(df, end_date=dates[-1], window_days=30)
    
    assert m is not None
    # Should capture growth from around day 9 to day 39
    assert m > 0.0


def test_compute_momentum_window_insufficient_data():
    """Test that None is returned when there's insufficient data."""
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]},
        index=dates,
    )

    m = compute_momentum_window(df, end_date=dates[-1], window_days=10)
    
    # Less than 5 data points in window
    assert m is None


def test_compute_momentum_window_empty():
    """Test that None is returned for empty dataframe."""
    df = pd.DataFrame()
    m = compute_momentum_window(df, end_date=pd.Timestamp("2020-01-01"), window_days=30)
    
    assert m is None


def test_compute_pre_earnings_stats_basic():
    """Test basic pre-earnings statistics computation."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0, 
                         105.0, 106.0, 107.0, 108.0, 109.0],
            "volume": [1000, 1100, 1200, 1300, 1400,
                      1500, 1600, 1700, 1800, 1900],
        },
        index=dates,
    )

    stats = compute_pre_earnings_stats(
        df, pre_start=dates[0], pre_end=dates[-1]
    )

    assert "pre_volatility_30d" in stats
    assert "pre_avg_volume_30d" in stats
    
    # Average volume should be 1450
    assert stats["pre_avg_volume_30d"] is not None
    assert np.isclose(stats["pre_avg_volume_30d"], 1450.0, atol=1.0)
    
    # Volatility should be positive (returns have some variance)
    assert stats["pre_volatility_30d"] is not None
    assert stats["pre_volatility_30d"] > 0.0


def test_compute_pre_earnings_stats_no_volume():
    """Test stats computation when volume column is missing."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0, 103.0, 104.0,
                      105.0, 106.0, 107.0, 108.0, 109.0]},
        index=dates,
    )

    stats = compute_pre_earnings_stats(
        df, pre_start=dates[0], pre_end=dates[-1]
    )

    assert stats["pre_volatility_30d"] is not None
    assert stats["pre_avg_volume_30d"] is None


def test_compute_pre_earnings_stats_insufficient_data():
    """Test that None is returned when there's insufficient data."""
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "adj_close": [100.0, 101.0, 102.0],
            "volume": [1000, 1100, 1200],
        },
        index=dates,
    )

    stats = compute_pre_earnings_stats(
        df, pre_start=dates[0], pre_end=dates[-1]
    )

    # Less than 5 data points
    assert stats["pre_volatility_30d"] is None
    assert stats["pre_avg_volume_30d"] is None


def test_compute_pre_earnings_stats_empty():
    """Test that None values are returned for empty dataframe."""
    df = pd.DataFrame()
    stats = compute_pre_earnings_stats(
        df,
        pre_start=pd.Timestamp("2020-01-01"),
        pre_end=pd.Timestamp("2020-01-31")
    )

    assert stats["pre_volatility_30d"] is None
    assert stats["pre_avg_volume_30d"] is None
