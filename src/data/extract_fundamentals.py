"""
Extract and pivot fundamental data from RAW_DATA.csv.

This module properly extracts the fundamental data which is stored in a
wide format (15 features per company as separate rows, quarters as columns).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Settings


logger = logging.getLogger(__name__)


def parse_feature_description(feature_desc: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse feature description to extract company name and feature name.
    
    Format: "COMPANY NAME - FEATURE NAME"
    Example: "AMAZON.COM - NET SALES OR REVENUES"
    
    Parameters
    ----------
    feature_desc : str
        Feature description from #NOM?.1 column
        
    Returns
    -------
    tuple[Optional[str], Optional[str]]
        (company_name, feature_name)
    """
    if pd.isna(feature_desc):
        return None, None
    
    parts = str(feature_desc).split(' - ', 1)  # Split on first ' - ' only
    if len(parts) >= 2:
        company = parts[0].strip()
        feature = parts[1].strip()
        return company, feature
    
    return None, None


def extract_fundamentals_from_raw_data(
    raw_data_path: Path,
    settings: Settings,
) -> pd.DataFrame:
    """
    Extract and pivot fundamental data from RAW_DATA.csv.
    
    The RAW_DATA.csv has a complex structure:
    - Each company has 15 rows (one per feature)
    - Quarters are in columns (Q1 2010, Q2 2010, ...)
    - Company name is embedded in the feature description
    
    This function:
    1. Parses company names from feature descriptions
    2. Pivots quarters from columns to rows
    3. Pivots features from rows to columns
    4. Returns one row per company-quarter with all features
    
    Parameters
    ----------
    raw_data_path : Path
        Path to RAW_DATA.csv
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - company_name
        - quarter (1, 2, 3, 4)
        - year
        - [15 fundamental feature columns]
    """
    logger.info(f"Loading RAW_DATA.csv from {raw_data_path}")
    
    # Load RAW_DATA.csv
    df = pd.read_csv(raw_data_path, sep=';', encoding='latin-1', low_memory=False)
    
    logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Parse company and feature names
    logger.info("  Parsing company and feature names...")
    parsed = df['#NOM?.1'].apply(lambda x: parse_feature_description(x))
    df['company_name'] = parsed.apply(lambda x: x[0])
    df['feature_name'] = parsed.apply(lambda x: x[1])
    
    # Remove rows without valid company/feature
    df = df[df['company_name'].notna() & df['feature_name'].notna()].copy()
    
    logger.info(f"  Found {df['company_name'].nunique()} unique companies")
    logger.info(f"  Found {df['feature_name'].nunique()} unique features")
    
    # CRITICAL: Extract the correct ISIN for each company
    # The ISIN CODE column is misaligned, so we need to find the correct ISIN
    # by looking at which ISIN appears most frequently for each company
    logger.info("  Extracting correct ISIN codes for each company...")
    company_to_isin = {}
    for company in df['company_name'].unique():
        company_rows = df[df['company_name'] == company]
        # Get the most common ISIN for this company
        isin_counts = company_rows['ISIN CODE'].value_counts()
        if len(isin_counts) > 0:
            company_to_isin[company] = isin_counts.index[0]
    
    # Add correct ISIN to dataframe
    df['isin_code'] = df['company_name'].map(company_to_isin)
    logger.info(f"  Mapped {len(company_to_isin)} companies to ISIN codes")
    
    # Get quarter columns (Q1 2010, Q2 2010, etc.)
    quarter_cols = [col for col in df.columns if col.startswith('Q')]
    logger.info(f"  Found {len(quarter_cols)} quarter columns")
    
    # Melt quarters from columns to rows
    logger.info("  Pivoting quarters from columns to rows...")
    # Use the corrected isin_code
    id_vars = ['company_name', 'feature_name', 'isin_code']
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=quarter_cols,
        var_name='quarter_str',
        value_name='value'
    )
    
    # Parse quarter string (e.g., "Q1 2010" -> quarter=1, year=2010)
    def parse_quarter(q_str):
        parts = q_str.split()
        if len(parts) == 2:
            quarter = int(parts[0][1])  # "Q1" -> 1
            year = int(parts[1])
            return quarter, year
        return None, None
    
    parsed_q = df_long['quarter_str'].apply(parse_quarter)
    df_long['quarter'] = parsed_q.apply(lambda x: x[0])
    df_long['year'] = parsed_q.apply(lambda x: x[1])
    
    # Remove invalid quarters
    df_long = df_long[df_long['quarter'].notna()].copy()
    
    logger.info(f"  After melting: {len(df_long):,} rows")
    
    # Clean values (convert comma decimals to dots, handle non-numeric)
    logger.info("  Cleaning numeric values...")
    
    def clean_value(val):
        if pd.isna(val):
            return np.nan
        # Convert to string and replace comma with dot
        val_str = str(val).replace(',', '.')
        try:
            return float(val_str)
        except:
            return np.nan
    
    df_long['value'] = df_long['value'].apply(clean_value)
    
    # Standardize feature names to snake_case
    logger.info("  Standardizing feature names...")
    
    def standardize_feature_name(name):
        if pd.isna(name):
            return name
        # Convert to lowercase and replace special chars with underscore
        name = str(name).lower()
        name = name.replace('-', '_').replace('/', '_').replace(' ', '_')
        name = name.replace('.', '').replace('&', 'and')
        # Remove multiple underscores
        while '__' in name:
            name = name.replace('__', '_')
        return name.strip('_')
    
    df_long['feature_name_std'] = df_long['feature_name'].apply(standardize_feature_name)
    
    # Pivot features from rows to columns
    logger.info("  Pivoting features from rows to columns...")
    # DON'T include isin_code in pivot - it's mostly null and will drop companies
    # We'll add correct ISINs later from earnings dates
    df_wide = df_long.pivot_table(
        index=['company_name', 'quarter', 'year'],
        columns='feature_name_std',
        values='value',
        aggfunc='first'  # Take first value if duplicates
    ).reset_index()
    
    logger.info(f"  Final shape: {len(df_wide):,} rows × {len(df_wide.columns)} columns")
    logger.info(f"  Companies after pivot: {df_wide['company_name'].nunique()}")
    logger.info(f"  Feature columns: {len([c for c in df_wide.columns if c not in ['company_name', 'quarter', 'year']])}")
    
    return df_wide


def merge_with_earnings_dates(
    fundamentals_df: pd.DataFrame,
    earnings_dates_df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """
    Merge fundamental data with earnings announcement dates.
    
    First adds ISIN codes from earnings dates, then merges on ISIN+quarter+year.
    
    Parameters
    ----------
    fundamentals_df : pd.DataFrame
        Fundamental data (one row per company-quarter)
    earnings_dates_df : pd.DataFrame
        Earnings dates from Quarter_1-4.csv with isin_code column
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame
        Merged data with earnings_date and isin_code columns
    """
    logger.info("Merging fundamental data with earnings dates...")
    
    # CRITICAL: Add ISIN codes from earnings dates to fundamentals
    # The ISIN in RAW_DATA is misaligned, so we get it from Quarter files instead
    logger.info("  Adding ISIN codes from earnings dates...")
    
    # Create company_name -> isin_code mapping from earnings dates
    company_to_isin = earnings_dates_df[['company_name', 'isin_code']].drop_duplicates()
    company_to_isin = company_to_isin.set_index('company_name')['isin_code'].to_dict()
    
    # Update isin_code in fundamentals (overwrite the incorrect ones from RAW_DATA)
    fundamentals_df['isin_code'] = fundamentals_df['company_name'].map(company_to_isin)
    
    # Count how many got ISINs
    has_isin = fundamentals_df['isin_code'].notna().sum()
    logger.info(f"  Mapped {has_isin:,} / {len(fundamentals_df):,} rows to ISIN codes")
    
    # Now merge on isin_code, quarter, year
    merged = fundamentals_df.merge(
        earnings_dates_df[['isin_code', 'quarter', 'year', 'earnings_date']],
        on=['isin_code', 'quarter', 'year'],
        how='left'
    )
    
    # Count how many have earnings dates
    has_date = merged['earnings_date'].notna().sum()
    logger.info(f"  Matched {has_date:,} / {len(merged):,} rows with earnings dates ({100*has_date/len(merged):.1f}%)")
    
    return merged


def add_ticker_mapping(
    df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """
    Add ticker symbols using the ticker mapping.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with company_name column
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame
        DataFrame with ticker column added
    """
    from src.data.ticker_mapping import add_ticker_column
    
    logger.info("Adding ticker symbols...")
    df_with_ticker = add_ticker_column(df, company_col='company_name')
    
    has_ticker = df_with_ticker['ticker'].notna().sum()
    logger.info(f"  Mapped {has_ticker:,} / {len(df_with_ticker):,} rows to tickers ({100*has_ticker/len(df_with_ticker):.1f}%)")
    
    return df_with_ticker
