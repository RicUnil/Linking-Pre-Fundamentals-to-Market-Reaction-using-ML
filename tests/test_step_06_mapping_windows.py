"""
Tests for Step 06: Earnings event to daily data window mapping.

This module contains unit tests for the temporal mapping functions that
compute pre- and post-earnings windows and flag data coverage.
"""

import pytest
import pandas as pd

from src.config import Settings
from src.data.mapping import compute_event_windows, flag_window_coverage


def test_compute_event_windows_adds_columns():
    """
    Test that compute_event_windows adds the expected window columns.
    
    This test verifies that:
    - All four window columns are added
    - Window boundaries are computed correctly
    - Pre-window ends one day before earnings date
    - Post-window starts one day after earnings date
    """
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "TICKER"
    settings.EARNINGS_DATE_COLUMN = "EARNINGS_DATE"
    
    df = pd.DataFrame({
        "TICKER": ["AAPL"],
        "EARNINGS_DATE": [pd.Timestamp("2020-02-15")],
    })
    
    out = compute_event_windows(df, settings)
    
    # Check that all window columns were added
    assert "pre_window_start" in out.columns
    assert "pre_window_end" in out.columns
    assert "post_window_start" in out.columns
    assert "post_window_end" in out.columns
    
    # Check specific values
    assert out.loc[0, "pre_window_start"] == pd.Timestamp("2020-01-16")  # 30 days before
    assert out.loc[0, "pre_window_end"] == pd.Timestamp("2020-02-14")    # 1 day before
    assert out.loc[0, "post_window_start"] == pd.Timestamp("2020-02-16")  # 1 day after
    assert out.loc[0, "post_window_end"] == pd.Timestamp("2020-03-16")    # 30 days after


def test_compute_event_windows_handles_multiple_events():
    """
    Test that compute_event_windows handles multiple earnings events.
    """
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "TICKER"
    settings.EARNINGS_DATE_COLUMN = "EARNINGS_DATE"
    
    df = pd.DataFrame({
        "TICKER": ["AAPL", "MSFT", "GOOGL"],
        "EARNINGS_DATE": [
            pd.Timestamp("2020-02-15"),
            pd.Timestamp("2020-03-20"),
            pd.Timestamp("2020-04-10"),
        ],
    })
    
    out = compute_event_windows(df, settings)
    
    # Check that all rows are processed
    assert len(out) == 3
    
    # Check that windows are different for each event
    assert out.loc[0, "pre_window_start"] != out.loc[1, "pre_window_start"]
    assert out.loc[1, "post_window_end"] != out.loc[2, "post_window_end"]


def test_compute_event_windows_drops_invalid_dates():
    """
    Test that compute_event_windows drops rows with invalid dates.
    """
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "TICKER"
    settings.EARNINGS_DATE_COLUMN = "EARNINGS_DATE"
    
    df = pd.DataFrame({
        "TICKER": ["AAPL", "MSFT", "GOOGL"],
        "EARNINGS_DATE": [
            pd.Timestamp("2020-02-15"),
            None,  # Invalid date
            pd.Timestamp("2020-04-10"),
        ],
    })
    
    out = compute_event_windows(df, settings)
    
    # Should drop the row with None date
    assert len(out) == 2
    assert "MSFT" not in out["TICKER"].values


def test_flag_window_coverage_uses_manifest():
    """
    Test that flag_window_coverage correctly uses the manifest.
    
    This test verifies that:
    - Coverage flags are added
    - Flags are True when manifest covers the windows
    - Flags are False when ticker is not in manifest
    """
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "TICKER"
    settings.EARNINGS_DATE_COLUMN = "EARNINGS_DATE"
    
    df = pd.DataFrame({
        "TICKER": ["AAPL"],
        "EARNINGS_DATE": [pd.Timestamp("2020-02-15")],
    })
    df = compute_event_windows(df, settings)
    
    # Manifest with sufficient coverage
    manifest = {
        "AAPL": {"start": "2019-12-01", "end": "2020-04-01", "rows": 100}
    }
    
    out = flag_window_coverage(df, manifest, settings)
    
    # Check that coverage columns were added
    assert "has_full_pre_window" in out.columns
    assert "has_full_post_window" in out.columns
    
    # Check that both windows are covered
    assert out.loc[0, "has_full_pre_window"] == True
    assert out.loc[0, "has_full_post_window"] == True


def test_flag_window_coverage_detects_insufficient_coverage():
    """
    Test that flag_window_coverage detects when coverage is insufficient.
    """
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "TICKER"
    settings.EARNINGS_DATE_COLUMN = "EARNINGS_DATE"
    
    df = pd.DataFrame({
        "TICKER": ["AAPL"],
        "EARNINGS_DATE": [pd.Timestamp("2020-02-15")],
    })
    df = compute_event_windows(df, settings)
    
    # Manifest with insufficient coverage (ends before post-window)
    manifest = {
        "AAPL": {"start": "2019-12-01", "end": "2020-02-20", "rows": 50}
    }
    
    out = flag_window_coverage(df, manifest, settings)
    
    # Pre-window should be covered
    assert out.loc[0, "has_full_pre_window"] == True
    
    # Post-window should NOT be covered (ends 2020-03-16, but manifest ends 2020-02-20)
    assert out.loc[0, "has_full_post_window"] == False


def test_flag_window_coverage_handles_missing_ticker():
    """
    Test that flag_window_coverage handles tickers not in manifest.
    """
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "TICKER"
    settings.EARNINGS_DATE_COLUMN = "EARNINGS_DATE"
    
    df = pd.DataFrame({
        "TICKER": ["AAPL", "TSLA"],
        "EARNINGS_DATE": [
            pd.Timestamp("2020-02-15"),
            pd.Timestamp("2020-03-10"),
        ],
    })
    df = compute_event_windows(df, settings)
    
    # Manifest only has AAPL
    manifest = {
        "AAPL": {"start": "2019-12-01", "end": "2020-04-01", "rows": 100}
    }
    
    out = flag_window_coverage(df, manifest, settings)
    
    # AAPL should be covered
    assert out.loc[0, "has_full_pre_window"] == True
    assert out.loc[0, "has_full_post_window"] == True
    
    # TSLA should NOT be covered (not in manifest)
    assert out.loc[1, "has_full_pre_window"] == False
    assert out.loc[1, "has_full_post_window"] == False


def test_flag_window_coverage_preserves_data():
    """
    Test that flag_window_coverage doesn't modify existing columns.
    """
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "TICKER"
    settings.EARNINGS_DATE_COLUMN = "EARNINGS_DATE"
    
    df = pd.DataFrame({
        "TICKER": ["AAPL"],
        "EARNINGS_DATE": [pd.Timestamp("2020-02-15")],
        "SOME_VALUE": [42],
    })
    df = compute_event_windows(df, settings)
    
    manifest = {
        "AAPL": {"start": "2019-12-01", "end": "2020-04-01", "rows": 100}
    }
    
    out = flag_window_coverage(df, manifest, settings)
    
    # Check that original data is preserved
    assert out.loc[0, "SOME_VALUE"] == 42
    assert out.loc[0, "TICKER"] == "AAPL"
