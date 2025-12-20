"""
Unit tests for fundamental feature engineering.

Tests the core logic of QoQ/YoY growth calculations and ratio computations.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.preprocessing.features import (
    add_fundamental_features,
    standardize_column_names,
    map_fundamental_columns,
)


def test_standardize_column_names():
    """Test that column names are properly standardized."""
    df = pd.DataFrame(
        {
            "NET SALES OR REVENUES": [100, 200],
            "EPS-INT SURP VALUE": [1.0, 2.0],
            "Total Assets": [500, 600],
        }
    )
    
    df_std = standardize_column_names(df)
    
    # Check that columns are lowercase and special chars replaced
    assert "net_sales_or_revenues" in df_std.columns
    assert "eps_int_surp_value" in df_std.columns
    assert "total_assets" in df_std.columns
    
    # Original should be unchanged
    assert "NET SALES OR REVENUES" in df.columns


def test_map_fundamental_columns():
    """Test that fundamental columns are correctly mapped."""
    std_columns = [
        "net_sales_or_revenues",
        "operating_income",
        "net_inc_before_extra_pfd_divs",
        "total_assets",
        "total_liabilities",
        "eps_int_surp_value",
    ]
    
    col_map = map_fundamental_columns(std_columns)
    
    # Check that key columns are mapped
    assert col_map['net_sales'] == "net_sales_or_revenues"
    assert col_map['operating_income'] == "operating_income"
    assert col_map['net_income'] == "net_inc_before_extra_pfd_divs"
    assert col_map['total_assets'] == "total_assets"
    assert col_map['total_liabilities'] == "total_liabilities"
    assert col_map['eps_surprise_value'] == "eps_int_surp_value"


def test_add_fundamental_features_computes_qoq_growth():
    """Test that QoQ growth is correctly computed."""
    settings = Settings(
        EARNINGS_TICKER_COLUMN="ticker",
        EARNINGS_DATE_COLUMN="earnings_date",
    )

    df = pd.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "earnings_date": [
                pd.Timestamp("2020-03-31"),
                pd.Timestamp("2020-06-30"),
                pd.Timestamp("2020-09-30"),
            ],
            "NET SALES OR REVENUES": [100.0, 110.0, 121.0],
            "OPERATING INCOME": [10.0, 11.0, 12.1],
            "NET INC BEFORE EXTRA/PFD DIVS": [8.0, 8.8, 9.68],
            "TOTAL ASSETS": [200.0, 210.0, 220.0],
            "TOTAL LIABILITIES": [100.0, 105.0, 110.0],
            "FUNDS FROM OPERATIONS": [15.0, 16.0, 17.0],
            "CAPITAL EXPENDITURES": [5.0, 5.0, 5.0],
        }
    )

    enriched = add_fundamental_features(df, settings)

    # Check revenue growth QoQ: 100 -> 110 -> 121  => +10%, +10%
    # Allow some tolerance for floating point
    qoq = enriched["f_revenue_growth_qoq"].values
    assert np.isnan(qoq[0])  # first quarter has no previous
    assert np.isclose(qoq[1], 0.10, atol=1e-6)
    assert np.isclose(qoq[2], 0.10, atol=1e-6)

    # Check operating margin: operating_income / net_sales
    op_margin = enriched["f_operating_margin"].values
    assert np.isclose(op_margin[0], 10.0 / 100.0, atol=1e-6)
    assert np.isclose(op_margin[1], 11.0 / 110.0, atol=1e-6)
    assert np.isclose(op_margin[2], 12.1 / 121.0, atol=1e-6)


def test_add_fundamental_features_computes_yoy_growth():
    """Test that YoY growth is correctly computed."""
    settings = Settings(
        EARNINGS_TICKER_COLUMN="ticker",
        EARNINGS_DATE_COLUMN="earnings_date",
    )

    # Create 5 quarters of data (need at least 5 to test YoY on 5th)
    df = pd.DataFrame(
        {
            "ticker": ["A"] * 5,
            "earnings_date": [
                pd.Timestamp("2020-03-31"),
                pd.Timestamp("2020-06-30"),
                pd.Timestamp("2020-09-30"),
                pd.Timestamp("2020-12-31"),
                pd.Timestamp("2021-03-31"),  # 4 quarters after first
            ],
            "NET SALES OR REVENUES": [100.0, 110.0, 121.0, 133.1, 120.0],  # Last is 20% more than first
        }
    )

    enriched = add_fundamental_features(df, settings)

    # Check YoY growth: compare quarter 5 to quarter 1
    yoy = enriched["f_revenue_growth_yoy"].values
    
    # First 4 quarters should be NaN (no data 4 quarters prior)
    assert np.isnan(yoy[0])
    assert np.isnan(yoy[1])
    assert np.isnan(yoy[2])
    assert np.isnan(yoy[3])
    
    # 5th quarter: 120 vs 100 = 20% growth
    assert np.isclose(yoy[4], 0.20, atol=1e-6)


def test_add_fundamental_features_computes_ratios():
    """Test that financial ratios are correctly computed."""
    settings = Settings(
        EARNINGS_TICKER_COLUMN="ticker",
        EARNINGS_DATE_COLUMN="earnings_date",
    )

    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "earnings_date": [pd.Timestamp("2020-03-31")],
            "NET SALES OR REVENUES": [100.0],
            "OPERATING INCOME": [10.0],
            "NET INC BEFORE EXTRA/PFD DIVS": [8.0],
            "TOTAL ASSETS": [200.0],
            "TOTAL LIABILITIES": [100.0],
            "COMMON SHAREHOLDERS' EQUITY": [100.0],
            "CASH": [20.0],
            "FUNDS FROM OPERATIONS": [15.0],
            "CAPITAL EXPENDITURES": [5.0],
        }
    )

    enriched = add_fundamental_features(df, settings)

    # Operating margin = 10 / 100 = 0.10
    assert np.isclose(enriched["f_operating_margin"].iloc[0], 0.10, atol=1e-6)
    
    # Net margin = 8 / 100 = 0.08
    assert np.isclose(enriched["f_net_margin"].iloc[0], 0.08, atol=1e-6)
    
    # Leverage = 100 / 200 = 0.50
    assert np.isclose(enriched["f_leverage"].iloc[0], 0.50, atol=1e-6)
    
    # Equity ratio = 100 / 200 = 0.50
    assert np.isclose(enriched["f_equity_ratio"].iloc[0], 0.50, atol=1e-6)
    
    # Cash to assets = 20 / 200 = 0.10
    assert np.isclose(enriched["f_cash_to_assets_ratio"].iloc[0], 0.10, atol=1e-6)
    
    # Cashflow proxy = 15 - 5 = 10
    assert np.isclose(enriched["f_cashflow_proxy_ffo_capex"].iloc[0], 10.0, atol=1e-6)


def test_add_fundamental_features_handles_multiple_tickers():
    """Test that growth calculations are done per ticker."""
    settings = Settings(
        EARNINGS_TICKER_COLUMN="ticker",
        EARNINGS_DATE_COLUMN="earnings_date",
    )

    df = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "earnings_date": [
                pd.Timestamp("2020-03-31"),
                pd.Timestamp("2020-06-30"),
                pd.Timestamp("2020-03-31"),
                pd.Timestamp("2020-06-30"),
            ],
            "NET SALES OR REVENUES": [100.0, 110.0, 200.0, 240.0],
        }
    )

    enriched = add_fundamental_features(df, settings)

    qoq = enriched["f_revenue_growth_qoq"].values
    
    # First quarter of each ticker should be NaN
    assert np.isnan(qoq[0])  # A Q1
    assert np.isnan(qoq[2])  # B Q1
    
    # Second quarter of each ticker should show growth
    assert np.isclose(qoq[1], 0.10, atol=1e-6)  # A: 110/100 - 1 = 10%
    assert np.isclose(qoq[3], 0.20, atol=1e-6)  # B: 240/200 - 1 = 20%


def test_add_fundamental_features_handles_missing_columns():
    """Test that missing fundamental columns are handled gracefully."""
    settings = Settings(
        EARNINGS_TICKER_COLUMN="ticker",
        EARNINGS_DATE_COLUMN="earnings_date",
    )

    # DataFrame with only minimal columns
    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "earnings_date": [pd.Timestamp("2020-03-31")],
            "NET SALES OR REVENUES": [100.0],
        }
    )

    enriched = add_fundamental_features(df, settings)

    # Revenue growth should be NaN (only one quarter)
    assert np.isnan(enriched["f_revenue_growth_qoq"].iloc[0])
    
    # Ratios requiring missing columns should be NaN
    assert np.isnan(enriched["f_operating_margin"].iloc[0])
    assert np.isnan(enriched["f_leverage"].iloc[0])
    
    # But the function should not crash
    assert len(enriched) == 1


def test_add_fundamental_features_preserves_original_columns():
    """Test that original columns are preserved."""
    settings = Settings(
        EARNINGS_TICKER_COLUMN="ticker",
        EARNINGS_DATE_COLUMN="earnings_date",
    )

    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "earnings_date": [pd.Timestamp("2020-03-31")],
            "NET SALES OR REVENUES": [100.0],
            "excess_return_30d": [0.05],  # Target from Step 07
        }
    )

    enriched = add_fundamental_features(df, settings)

    # Original columns should still be there
    assert "ticker" in enriched.columns
    assert "earnings_date" in enriched.columns
    assert "NET SALES OR REVENUES" in enriched.columns
    assert "excess_return_30d" in enriched.columns
    
    # And original values should be unchanged
    assert enriched["excess_return_30d"].iloc[0] == 0.05
