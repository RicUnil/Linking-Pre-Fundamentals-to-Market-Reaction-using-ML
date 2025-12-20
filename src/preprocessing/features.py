"""
Fundamental feature engineering for earnings events.

This module provides utilities to compute accounting-based features
from quarterly financial data, including:
- EPS surprise metrics
- Revenue and income growth rates (QoQ and YoY)
- Profitability margins
- Leverage and liquidity ratios
- Cashflow proxies
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import logging
import numpy as np
import pandas as pd

from src.config import Settings


logger = logging.getLogger(__name__)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with standardized snake_case column names.

    - Lowercase all column names.
    - Replace spaces with underscores.
    - Replace special characters like '-' and '/' with underscores.

    This is helpful for working with Capital IQ-style column names such as
    'NET SALES OR REVENUES' or 'EPS-INT SURP VALUE'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potentially non-standard column names.

    Returns
    -------
    pd.DataFrame
        Copy of the DataFrame with standardized column names.
    """
    df_copy = df.copy()
    
    # Standardize column names
    df_copy.columns = (
        df_copy.columns
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    
    return df_copy


def map_fundamental_columns(std_columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Map standardized column names to canonical fundamental keys.

    Parameters
    ----------
    std_columns : list of str
        List of standardized column names.

    Returns
    -------
    dict
        Mapping with keys like:
            'eps_surprise_value'
            'eps_surprise_pct'
            'net_sales'
            'operating_income'
            'net_income'
            'total_assets'
            'total_liabilities'
            'equity'
            'cash'
            'funds_from_operations'
            'capital_expenditures'
        Values are the matching column names (or None if not found).
    """
    # Define mapping patterns
    patterns = {
        'eps_surprise_value': ['eps_int_surp_value', 'eps_surp_value'],
        'eps_surprise_pct': ['eps_int_surp_pdiff', 'eps_surp_pdiff', 'eps_surprise_pct'],
        'net_sales': ['net_sales_or_revenues', 'net_sales', 'revenues'],
        'operating_income': ['operating_income', 'op_income'],
        'net_income': ['net_inc_before_extra_pfd_divs', 'net_income', 'net_inc'],
        'total_assets': ['total_assets', 'assets'],
        'total_liabilities': ['total_liabilities', 'liabilities'],
        'equity': ['common_shareholders__equity', 'common_shareholders_equity', 'shareholders_equity', 'equity'],
        'cash': ['cash'],
        'funds_from_operations': ['funds_from_operations', 'ffo'],
        'capital_expenditures': ['capital_expenditures', 'capex'],
    }
    
    # Initialize result
    result = {}
    
    # For each canonical key, find the first matching column
    for key, possible_names in patterns.items():
        matched = None
        for pattern in possible_names:
            for col in std_columns:
                if pattern in col:
                    matched = col
                    break
            if matched:
                break
        
        result[key] = matched
        
        if matched is None:
            logger.warning(f"Could not map fundamental column: {key}")
    
    return result


def add_fundamental_features(
    earnings_with_targets: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """
    Add fundamental accounting-based features to the earnings dataset.

    Features include (when data is available):
    - eps_surprise_value
    - eps_surprise_pct
    - revenue_growth_qoq
    - revenue_growth_yoy
    - operating_margin
    - net_margin
    - net_income_change_qoq
    - net_income_change_yoy
    - cashflow_proxy_ffo_capex
    - leverage
    - equity_ratio (optional)
    - cash_to_assets_ratio (optional)

    Parameters
    ----------
    earnings_with_targets : pd.DataFrame
        DataFrame from Step 07 containing at least ticker, earnings date,
        and raw fundamental columns.
    settings : Settings
        Global project settings, used to know ticker and date columns.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with additional feature columns.
    """
    logger.info("Adding fundamental features to earnings dataset...")
    
    # Work on a copy
    df = earnings_with_targets.copy()
    
    # Get ticker and date columns
    ticker_col = settings.EARNINGS_TICKER_COLUMN
    date_col = settings.EARNINGS_DATE_COLUMN
    
    logger.info(f"  Using ticker column: {ticker_col}")
    logger.info(f"  Using date column: {date_col}")
    
    # Check if columns are already standardized (lowercase, underscores)
    # If most columns are already lowercase, skip standardization
    lowercase_cols = sum(1 for col in df.columns if col.islower())
    already_standardized = lowercase_cols > len(df.columns) * 0.8
    
    if already_standardized:
        logger.info("  Columns appear already standardized, using as-is")
        df_std = df.copy()
        ticker_col_std = ticker_col
        date_col_std = date_col
    else:
        logger.info("  Standardizing column names...")
        df_std = standardize_column_names(df)
        # Also standardize the ticker and date column names
        ticker_col_std = ticker_col.lower().replace(r"[^0-9a-z]+", "_").strip("_")
        date_col_std = date_col.lower().replace(r"[^0-9a-z]+", "_").strip("_")
    
    # Map fundamental columns
    col_map = map_fundamental_columns(list(df_std.columns))
    
    logger.info("  Fundamental column mapping:")
    for key, val in col_map.items():
        if val:
            logger.info(f"    {key}: {val}")
    
    # Initialize feature columns with NaN
    feature_columns = [
        'f_eps_surprise_value',
        'f_eps_surprise_pct',
        'f_revenue_growth_qoq',
        'f_revenue_growth_yoy',
        'f_operating_margin',
        'f_net_margin',
        'f_net_income_change_qoq',
        'f_net_income_change_yoy',
        'f_cashflow_proxy_ffo_capex',
        'f_leverage',
        'f_equity_ratio',
        'f_cash_to_assets_ratio',
        'f_roa',  # Return on Assets
        'f_roe',  # Return on Equity
        'f_asset_turnover',  # Asset Turnover
    ]
    
    for col in feature_columns:
        df[col] = np.nan
    
    # ========================================================================
    # 1. DIRECT COPY FEATURES (no computation needed)
    # ========================================================================
    
    # EPS surprise value
    if col_map['eps_surprise_value']:
        df['f_eps_surprise_value'] = df_std[col_map['eps_surprise_value']]
        logger.info(f"  ✓ f_eps_surprise_value: {df['f_eps_surprise_value'].notna().sum()} non-null")
    
    # EPS surprise percentage
    if col_map['eps_surprise_pct']:
        df['f_eps_surprise_pct'] = df_std[col_map['eps_surprise_pct']]
        logger.info(f"  ✓ f_eps_surprise_pct: {df['f_eps_surprise_pct'].notna().sum()} non-null")
    
    # ========================================================================
    # 2. RATIO FEATURES (current period only)
    # ========================================================================
    
    # Helper function to compute ratios safely (only where both values are non-zero)
    def safe_ratio(numerator, denominator):
        """Compute ratio only where both values are non-null and non-zero."""
        mask = (numerator.notna() & (numerator != 0) & 
                denominator.notna() & (denominator != 0))
        result = pd.Series(np.nan, index=numerator.index)
        result[mask] = numerator[mask] / denominator[mask]
        return result
    
    # Operating margin = operating_income / net_sales
    if col_map['operating_income'] and col_map['net_sales']:
        operating_income = df_std[col_map['operating_income']]
        net_sales = df_std[col_map['net_sales']]
        df['f_operating_margin'] = safe_ratio(operating_income, net_sales)
        logger.info(f"  ✓ f_operating_margin: {df['f_operating_margin'].notna().sum()} non-null")
    
    # Net margin = net_income / net_sales
    if col_map['net_income'] and col_map['net_sales']:
        net_income = df_std[col_map['net_income']]
        net_sales = df_std[col_map['net_sales']]
        df['f_net_margin'] = safe_ratio(net_income, net_sales)
        logger.info(f"  ✓ f_net_margin: {df['f_net_margin'].notna().sum()} non-null")
    
    # Leverage = total_liabilities / total_assets
    if col_map['total_liabilities'] and col_map['total_assets']:
        total_liabilities = df_std[col_map['total_liabilities']]
        total_assets = df_std[col_map['total_assets']]
        df['f_leverage'] = safe_ratio(total_liabilities, total_assets)
        logger.info(f"  ✓ f_leverage: {df['f_leverage'].notna().sum()} non-null")
    
    # Equity ratio = equity / total_assets
    if col_map['equity'] and col_map['total_assets']:
        equity = df_std[col_map['equity']]
        total_assets = df_std[col_map['total_assets']]
        df['f_equity_ratio'] = safe_ratio(equity, total_assets)
        logger.info(f"  ✓ f_equity_ratio: {df['f_equity_ratio'].notna().sum()} non-null")
    
    # Cash to assets ratio = cash / total_assets
    if col_map['cash'] and col_map['total_assets']:
        cash = df_std[col_map['cash']]
        total_assets = df_std[col_map['total_assets']]
        df['f_cash_to_assets_ratio'] = safe_ratio(cash, total_assets)
        logger.info(f"  ✓ f_cash_to_assets_ratio: {df['f_cash_to_assets_ratio'].notna().sum()} non-null")
    
    # Cashflow proxy = FFO - CAPEX (only where both are non-zero)
    if col_map['funds_from_operations'] and col_map['capital_expenditures']:
        ffo = df_std[col_map['funds_from_operations']]
        capex = df_std[col_map['capital_expenditures']]
        mask = (ffo.notna() & (ffo != 0) & capex.notna() & (capex != 0))
        df['f_cashflow_proxy_ffo_capex'] = np.nan
        df.loc[mask, 'f_cashflow_proxy_ffo_capex'] = ffo[mask] - capex[mask]
        logger.info(f"  ✓ f_cashflow_proxy_ffo_capex: {df['f_cashflow_proxy_ffo_capex'].notna().sum()} non-null")
    
    # Additional ratio: Return on Assets (ROA) = net_income / total_assets
    if col_map['net_income'] and col_map['total_assets']:
        net_income = df_std[col_map['net_income']]
        total_assets = df_std[col_map['total_assets']]
        df['f_roa'] = safe_ratio(net_income, total_assets)
        logger.info(f"  ✓ f_roa: {df['f_roa'].notna().sum()} non-null")
    
    # Additional ratio: Return on Equity (ROE) = net_income / equity
    if col_map['net_income'] and col_map['equity']:
        net_income = df_std[col_map['net_income']]
        equity = df_std[col_map['equity']]
        df['f_roe'] = safe_ratio(net_income, equity)
        logger.info(f"  ✓ f_roe: {df['f_roe'].notna().sum()} non-null")
    
    # Additional ratio: Asset turnover = net_sales / total_assets
    if col_map['net_sales'] and col_map['total_assets']:
        net_sales = df_std[col_map['net_sales']]
        total_assets = df_std[col_map['total_assets']]
        df['f_asset_turnover'] = safe_ratio(net_sales, total_assets)
        logger.info(f"  ✓ f_asset_turnover: {df['f_asset_turnover'].notna().sum()} non-null")
    
    # ========================================================================
    # 3. GROWTH FEATURES (require sorting and groupby)
    # ========================================================================
    
    logger.info("  Computing growth features (QoQ and YoY)...")
    
    # Sort by ticker and date
    df_sorted = df_std.sort_values(by=[ticker_col_std, date_col_std]).copy()
    
    # Revenue growth QoQ and YoY
    if col_map['net_sales']:
        net_sales_series = df_sorted[col_map['net_sales']]
        
        # Group by ticker
        grouped = df_sorted.groupby(ticker_col_std, sort=False)
        
        # QoQ: compare to previous quarter (shift 1)
        revenue_qoq = grouped[col_map['net_sales']].pct_change(periods=1, fill_method=None)
        
        # YoY: compare to 4 quarters ago (shift 4)
        revenue_yoy = grouped[col_map['net_sales']].pct_change(periods=4, fill_method=None)
        
        # Align back to original index
        df.loc[revenue_qoq.index, 'f_revenue_growth_qoq'] = revenue_qoq.values
        df.loc[revenue_yoy.index, 'f_revenue_growth_yoy'] = revenue_yoy.values
        
        logger.info(f"  ✓ f_revenue_growth_qoq: {df['f_revenue_growth_qoq'].notna().sum()} non-null")
        logger.info(f"  ✓ f_revenue_growth_yoy: {df['f_revenue_growth_yoy'].notna().sum()} non-null")
    
    # Net income change QoQ and YoY
    if col_map['net_income']:
        grouped = df_sorted.groupby(ticker_col_std, sort=False)
        
        # QoQ change (absolute difference)
        net_income_series = df_sorted[col_map['net_income']]
        net_income_qoq = grouped[col_map['net_income']].diff(periods=1)
        
        # YoY change (absolute difference)
        net_income_yoy = grouped[col_map['net_income']].diff(periods=4)
        
        # Align back to original index
        df.loc[net_income_qoq.index, 'f_net_income_change_qoq'] = net_income_qoq.values
        df.loc[net_income_yoy.index, 'f_net_income_change_yoy'] = net_income_yoy.values
        
        logger.info(f"  ✓ f_net_income_change_qoq: {df['f_net_income_change_qoq'].notna().sum()} non-null")
        logger.info(f"  ✓ f_net_income_change_yoy: {df['f_net_income_change_yoy'].notna().sum()} non-null")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("  Fundamental features added successfully")
    logger.info(f"  Total features added: {len(feature_columns)}")
    
    # Count how many features have at least some non-null values
    features_with_data = sum(1 for col in feature_columns if df[col].notna().sum() > 0)
    logger.info(f"  Features with data: {features_with_data}/{len(feature_columns)}")
    
    return df


# ============================================================================
# MARKET-BASED FEATURE ENGINEERING (Step 09)
# ============================================================================


def _select_price_column(daily_df: pd.DataFrame) -> str:
    """
    Choose which price column to use for market feature computation.

    Preference order:
    1. 'adj_close'
    2. 'close'

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily OHLCV data for a single ticker.

    Returns
    -------
    str
        Name of the selected price column.

    Raises
    ------
    ValueError
        If neither 'adj_close' nor 'close' is present.
    """
    cols = {c.lower(): c for c in daily_df.columns}
    if "adj_close" in cols:
        return cols["adj_close"]
    if "close" in cols:
        return cols["close"]
    raise ValueError("Neither 'adj_close' nor 'close' found in daily dataframe columns.")


def compute_daily_returns(daily_df: pd.DataFrame) -> pd.Series:
    """
    Compute simple daily returns from a daily OHLCV dataframe.

    Returns
    -------
    pd.Series
        Daily returns aligned with the dataframe index.
    """
    if daily_df.empty:
        return pd.Series(dtype="float64")

    df = daily_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    price_col = _select_price_column(df)
    returns = df[price_col].pct_change()
    return returns


def compute_momentum_window(
    daily_df: pd.DataFrame,
    end_date: pd.Timestamp,
    window_days: int,
) -> Optional[float]:
    """
    Compute simple price momentum over the given calendar window ending at end_date.

    Momentum = (price_end / price_start) - 1 over the last `window_days` days.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily OHLCV data indexed by datetime.
    end_date : pd.Timestamp
        End date of the window (typically pre_window_end).
    window_days : int
        Number of calendar days to look back.

    Returns
    -------
    Optional[float]
        Momentum value, or None if there is insufficient data.
    """
    if daily_df.empty:
        return None

    df = daily_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    start_date = end_date - pd.Timedelta(days=window_days)
    window = df.loc[(df.index >= start_date) & (df.index <= end_date)]

    if window.shape[0] < 5:
        return None

    price_col = _select_price_column(window)
    price_start = window[price_col].iloc[0]
    price_end = window[price_col].iloc[-1]

    if pd.isna(price_start) or pd.isna(price_end):
        return None

    return float(price_end / price_start - 1.0)


def compute_pre_earnings_stats(
    daily_df: pd.DataFrame,
    pre_start: pd.Timestamp,
    pre_end: pd.Timestamp,
) -> Dict[str, Optional[float]]:
    """
    Compute pre-earnings statistics (volatility, avg volume) over [pre_start, pre_end].

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily OHLCV data indexed by datetime.
    pre_start : pd.Timestamp
        Start of pre-earnings window (J-30).
    pre_end : pd.Timestamp
        End of pre-earnings window (J-1).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'pre_volatility_30d' : std of daily returns
        - 'pre_avg_volume_30d' : mean volume
    """
    result: Dict[str, Optional[float]] = {
        "pre_volatility_30d": None,
        "pre_avg_volume_30d": None,
    }

    if daily_df.empty:
        return result

    df = daily_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    window = df.loc[(df.index >= pre_start) & (df.index <= pre_end)]
    if window.shape[0] < 5:
        return result

    # Volatility from daily returns
    rets = compute_daily_returns(window)
    if rets.notna().sum() >= 3:
        result["pre_volatility_30d"] = float(rets.std())

    # Average volume
    vol_col = None
    for candidate in ["volume", "vol"]:
        if candidate in window.columns:
            vol_col = candidate
            break
    if vol_col is not None:
        result["pre_avg_volume_30d"] = float(window[vol_col].mean())

    return result


def add_market_features(
    earnings_df: pd.DataFrame,
    settings: Settings,
    daily_data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Add market-based features for each earnings event using daily price/volume
    data and pre-earnings windows.

    Features include (when data is available):
    - stock_momentum_1m   (~30 calendar days pre-earnings)
    - stock_momentum_3m   (~90 days)
    - stock_momentum_6m   (~180 days)
    - pre_volatility_30d  (std of daily returns over [J-30, J-1])
    - pre_avg_volume_30d
    - spy_pre_return_30d  (SPY return over [J-30, J-1])

    Parameters
    ----------
    earnings_df : pd.DataFrame
        Earnings dataset after Step 08, with pre_window_* columns and targets.
    settings : Settings
        Global project settings.
    daily_data_dir : Path, optional
        Directory containing '{TICKER}_daily.parquet' files from Step 05.
        Defaults to results/step_05/.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with additional market feature columns.
    """
    logger.info("Adding market-based features to earnings dataset...")
    
    if daily_data_dir is None:
        daily_data_dir = settings.RESULTS_DIR / "step_05"

    df = earnings_df.copy()
    ticker_col = settings.EARNINGS_TICKER_COLUMN
    pre_start_col = "pre_window_start"
    pre_end_col = "pre_window_end"

    # Try to load consolidated data first
    consolidated_path = daily_data_dir / "cache" / "all_daily_data.parquet"
    
    if consolidated_path.exists():
        logger.info(f"  Loading consolidated daily data from {consolidated_path}")
        all_daily_df = pd.read_parquet(consolidated_path)
        
        # Ensure date column is datetime and set as index
        if 'date' in all_daily_df.columns:
            all_daily_df['date'] = pd.to_datetime(all_daily_df['date'])
            all_daily_df = all_daily_df.set_index('date')
        
        # Extract SPY data
        spy_df = all_daily_df[all_daily_df['ticker'] == settings.SPY_TICKER].copy()
        spy_df = spy_df.sort_index()
        
        logger.info(f"  ✓ Loaded consolidated data: {all_daily_df['ticker'].nunique()} tickers")
        logger.info(f"  ✓ Loaded SPY data: {len(spy_df):,} days")
        
        use_consolidated = True
    else:
        # Fall back to individual files
        logger.info(f"  Consolidated data not found, using individual ticker files")
        spy_path = daily_data_dir / f"{settings.SPY_TICKER}_daily.parquet"
        
        if not spy_path.exists():
            logger.error(f"SPY data not found at {spy_path}")
            raise FileNotFoundError(f"SPY data not found at {spy_path}")
        
        spy_df = pd.read_parquet(spy_path)
        logger.info(f"  ✓ Loaded SPY data: {len(spy_df):,} days")
        
        use_consolidated = False
        all_daily_df = None

    # Prepare new columns
    df["stock_momentum_1m"] = np.nan
    df["stock_momentum_3m"] = np.nan
    df["stock_momentum_6m"] = np.nan
    df["pre_volatility_30d"] = np.nan
    df["pre_avg_volume_30d"] = np.nan
    df["spy_pre_return_30d"] = np.nan

    total_rows = df.shape[0]
    non_na_1m = 0
    non_na_3m = 0
    non_na_6m = 0
    non_na_vol = 0
    non_na_avg_vol = 0
    non_na_spy = 0
    missing_tickers = set()

    logger.info(f"  Processing {total_rows:,} events...")
    
    # Filter to events with full pre-window coverage
    valid_events = df[df["has_full_pre_window"] == True].copy()
    logger.info(f"  Events with full pre-window: {len(valid_events):,}")
    
    # Process by ticker to minimize data loading
    unique_tickers = valid_events[ticker_col].unique()
    logger.info(f"  Unique tickers to process: {len(unique_tickers)}")
    
    # Cache for ticker data
    ticker_cache = {}
    
    for i, ticker in enumerate(unique_tickers):
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(unique_tickers)} tickers processed")
        
        if pd.isna(ticker):
            continue
        
        # Load ticker data once per ticker
        if use_consolidated:
            ticker_df = all_daily_df[all_daily_df['ticker'] == ticker].copy()
            if len(ticker_df) == 0:
                if ticker not in missing_tickers:
                    missing_tickers.add(ticker)
                continue
            ticker_df = ticker_df.sort_index()
        else:
            ticker_path = daily_data_dir / f"{ticker}_daily.parquet"
            if not ticker_path.exists():
                if ticker not in missing_tickers:
                    missing_tickers.add(ticker)
                continue
            try:
                ticker_df = pd.read_parquet(ticker_path)
            except Exception:
                continue
        
        # Get all events for this ticker
        ticker_events = valid_events[valid_events[ticker_col] == ticker]
        
        for idx, row in ticker_events.iterrows():
            pre_start = row[pre_start_col]
            pre_end = row[pre_end_col]
            
            if pd.isna(pre_start) or pd.isna(pre_end):
                continue
            
            # Momentum windows
            m1 = compute_momentum_window(ticker_df, end_date=pre_end, window_days=30)
            m3 = compute_momentum_window(ticker_df, end_date=pre_end, window_days=90)
            m6 = compute_momentum_window(ticker_df, end_date=pre_end, window_days=180)
            
            if m1 is not None:
                df.at[idx, "stock_momentum_1m"] = m1
                non_na_1m += 1
            if m3 is not None:
                df.at[idx, "stock_momentum_3m"] = m3
                non_na_3m += 1
            if m6 is not None:
                df.at[idx, "stock_momentum_6m"] = m6
                non_na_6m += 1
            
            # Pre-earnings stats
            stats = compute_pre_earnings_stats(ticker_df, pre_start, pre_end)
            if stats["pre_volatility_30d"] is not None:
                df.at[idx, "pre_volatility_30d"] = stats["pre_volatility_30d"]
                non_na_vol += 1
            if stats["pre_avg_volume_30d"] is not None:
                df.at[idx, "pre_avg_volume_30d"] = stats["pre_avg_volume_30d"]
                non_na_avg_vol += 1
            
            # SPY pre-earnings return
            window_days = (pre_end - pre_start).days
            spy_ret = compute_momentum_window(spy_df, end_date=pre_end, window_days=window_days)
            if spy_ret is not None:
                df.at[idx, "spy_pre_return_30d"] = spy_ret
                non_na_spy += 1

    logger.info(f"\n  Market features computed:")
    logger.info(f"    stock_momentum_1m: {non_na_1m:,} non-null ({100*non_na_1m/total_rows:.1f}%)")
    logger.info(f"    stock_momentum_3m: {non_na_3m:,} non-null ({100*non_na_3m/total_rows:.1f}%)")
    logger.info(f"    stock_momentum_6m: {non_na_6m:,} non-null ({100*non_na_6m/total_rows:.1f}%)")
    logger.info(f"    pre_volatility_30d: {non_na_vol:,} non-null ({100*non_na_vol/total_rows:.1f}%)")
    logger.info(f"    pre_avg_volume_30d: {non_na_avg_vol:,} non-null ({100*non_na_avg_vol/total_rows:.1f}%)")
    logger.info(f"    spy_pre_return_30d: {non_na_spy:,} non-null ({100*non_na_spy/total_rows:.1f}%)")
    
    if missing_tickers:
        logger.warning(f"  ⚠ Missing daily data for {len(missing_tickers)} tickers")

    return df
