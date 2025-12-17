"""
Comprehensive data loader for all CSV files.

This module extracts and combines ALL data from:
- Quarter_1.csv, Quarter_2.csv, Quarter_3.csv, Quarter_4.csv (earnings dates)
- RAW_DATA.csv (all financial features)

The goal is to create a complete dataset with all companies, all features,
and all quarters, preserving N/A values where data is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from src.config import Settings


logger = logging.getLogger(__name__)


def clean_numeric_value(value: str) -> float:
    """
    Clean and convert numeric values from CSV format.
    
    Handles:
    - Comma as decimal separator (European format)
    - 'NA', 'N/A', empty strings -> NaN
    - Whitespace
    
    Parameters
    ----------
    value : str
        Raw value from CSV
        
    Returns
    -------
    float
        Cleaned numeric value or NaN
    """
    if pd.isna(value) or value == '':
        return np.nan
    
    # Convert to string and clean
    value_str = str(value).strip().upper()
    
    # Handle NA values
    if value_str in ['NA', 'N/A', '#N/A', 'NAN', '']:
        return np.nan
    
    # Replace comma with dot for decimal separator
    value_str = value_str.replace(',', '.')
    
    try:
        return float(value_str)
    except (ValueError, AttributeError):
        return np.nan


def extract_company_info(name_field: str) -> Tuple[str, str]:
    """
    Extract company name and feature name from the name field.
    
    Parameters
    ----------
    name_field : str
        Full name field (e.g., "AMAZON.COM - EPS-INT SURP VALUE")
        
    Returns
    -------
    Tuple[str, str]
        (company_name, feature_name)
    """
    if pd.isna(name_field):
        return "", ""
    
    name_str = str(name_field).strip()
    
    # Split on " - " to separate company from feature
    if " - " in name_str:
        parts = name_str.split(" - ", 1)
        company = parts[0].strip()
        feature = parts[1].strip() if len(parts) > 1 else ""
        return company, feature
    else:
        # If no separator, treat entire string as company name
        return name_str, ""


def load_quarterly_earnings_dates_comprehensive(
    quarter: int,
    settings: Settings,
) -> pd.DataFrame:
    """
    Load earnings dates from a quarterly file (comprehensive version).
    
    Parameters
    ----------
    quarter : int
        Quarter number (1, 2, 3, or 4)
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with earnings dates
    """
    file_path = settings.DATA_DIR / f"Quarter_{quarter}.csv"
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return pd.DataFrame()
    
    logger.info(f"  Loading Quarter {quarter} from {file_path.name}")
    
    try:
        # Load with error handling for malformed files
        df = pd.read_csv(file_path, sep=';', low_memory=False, encoding='utf-8')
        
        # Check if file has expected structure
        if '#NOM?' not in df.columns or 'ISIN CODE' not in df.columns:
            logger.warning(f"  Unexpected structure in Quarter_{quarter}.csv")
            return pd.DataFrame()
        
        logger.info(f"    Raw shape: {df.shape}")
        
        # Extract company names
        df['company_name'] = df['#NOM?'].apply(lambda x: extract_company_info(str(x))[0] if pd.notna(x) else "")
        
        # Filter out empty company names
        df = df[df['company_name'] != ''].copy()
        
        # Get year columns for this quarter
        year_columns = [col for col in df.columns if col.startswith(f'Q{quarter}')]
        
        logger.info(f"    Companies: {len(df)}, Year columns: {len(year_columns)}")
        
        # Convert to long format
        records = []
        
        for idx, row in df.iterrows():
            company_name = row['company_name']
            isin_code = row.get('ISIN CODE', '')
            
            for year_col in year_columns:
                date_str = row[year_col]
                
                # Parse date (DD.MM.YY format)
                if pd.notna(date_str) and str(date_str).strip() != '':
                    try:
                        # Extract year from column name
                        year = int(year_col.split()[-1])
                        
                        # Parse DD.MM.YY
                        parts = str(date_str).strip().split('.')
                        if len(parts) == 3:
                            day = int(parts[0])
                            month = int(parts[1])
                            year_suffix = int(parts[2])
                            
                            # Determine full year
                            full_year = (year // 100) * 100 + year_suffix
                            
                            earnings_date = pd.Timestamp(year=full_year, month=month, day=day)
                            
                            records.append({
                                'company_name': company_name,
                                'isin_code': isin_code,
                                'quarter': quarter,
                                'year': year,
                                'earnings_date': earnings_date
                            })
                    except (ValueError, IndexError):
                        continue
        
        result_df = pd.DataFrame(records)
        logger.info(f"    ✓ Extracted {len(result_df)} earnings dates")
        
        return result_df
        
    except Exception as e:
        logger.error(f"  Error loading Quarter_{quarter}.csv: {str(e)}")
        return pd.DataFrame()


def load_raw_data_features(settings: Settings) -> pd.DataFrame:
    """
    Load all features from RAW_DATA.csv.
    
    This file contains multiple features per company in a complex format.
    Each company may have multiple rows (one per feature).
    
    Parameters
    ----------
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with all features
    """
    file_path = settings.DATA_DIR / "RAW_DATA.csv"
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading RAW_DATA.csv from {file_path.name}")
    
    try:
        # Load the file
        df = pd.read_csv(file_path, sep=';', low_memory=False, encoding='utf-8')
        
        logger.info(f"  Raw shape: {df.shape}")
        logger.info(f"  Columns: {df.columns.tolist()[:10]}...")
        
        # The structure appears to be:
        # #NOM? (ID), NAME (company), ISIN CODE, empty col, #NOM?.1 (feature description), Q1 2010, Q2 2010, ...
        
        records = []
        
        for idx, row in df.iterrows():
            # Extract company info
            company_id = row.get('#NOM?', '')
            company_name = row.get('NAME', '')
            isin_code = row.get('ISIN CODE', '')
            
            # Extract feature description
            feature_desc = row.get('#NOM?.1', '')
            
            if pd.isna(company_name) or company_name == '':
                continue
            
            # Extract company name and feature name
            if pd.notna(feature_desc):
                _, feature_name = extract_company_info(str(feature_desc))
            else:
                feature_name = ""
            
            # Get all quarter columns
            quarter_cols = [col for col in df.columns if col.startswith('Q')]
            
            for quarter_col in quarter_cols:
                value = row[quarter_col]
                
                # Parse quarter and year from column name (e.g., "Q1 2010")
                try:
                    parts = quarter_col.split()
                    if len(parts) == 2:
                        quarter_str = parts[0]  # "Q1"
                        year = int(parts[1])     # 2010
                        quarter = int(quarter_str[1])  # 1
                        
                        # Clean the value
                        numeric_value = clean_numeric_value(value)
                        
                        records.append({
                            'company_id': company_id,
                            'company_name': str(company_name).strip(),
                            'isin_code': isin_code,
                            'feature_name': feature_name,
                            'quarter': quarter,
                            'year': year,
                            'value': numeric_value
                        })
                except (ValueError, IndexError):
                    continue
        
        result_df = pd.DataFrame(records)
        logger.info(f"  ✓ Extracted {len(result_df)} feature values")
        logger.info(f"  Unique companies: {result_df['company_name'].nunique()}")
        logger.info(f"  Unique features: {result_df['feature_name'].nunique()}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"  Error loading RAW_DATA.csv: {str(e)}")
        return pd.DataFrame()


def combine_all_data(settings: Settings) -> pd.DataFrame:
    """
    Combine all data from quarterly files and RAW_DATA.csv.
    
    Parameters
    ----------
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame
        Comprehensive dataset with all companies, features, and earnings dates
    """
    logger.info("=" * 70)
    logger.info("LOADING COMPREHENSIVE DATASET")
    logger.info("=" * 70)
    
    # Load earnings dates from all quarters
    logger.info("\n1. Loading earnings dates from quarterly files...")
    logger.info("-" * 70)
    
    all_earnings = []
    for quarter in [1, 2, 3, 4]:
        quarter_df = load_quarterly_earnings_dates_comprehensive(quarter, settings)
        if not quarter_df.empty:
            all_earnings.append(quarter_df)
    
    if all_earnings:
        earnings_df = pd.concat(all_earnings, ignore_index=True)
        logger.info(f"\n✓ Total earnings dates: {len(earnings_df):,}")
        logger.info(f"  Unique companies: {earnings_df['company_name'].nunique()}")
        logger.info(f"  Date range: {earnings_df['earnings_date'].min().date()} to {earnings_df['earnings_date'].max().date()}")
    else:
        earnings_df = pd.DataFrame()
        logger.warning("No earnings dates loaded")
    
    # Load features from RAW_DATA.csv
    logger.info("\n2. Loading features from RAW_DATA.csv...")
    logger.info("-" * 70)
    
    features_df = load_raw_data_features(settings)
    
    if features_df.empty:
        logger.warning("No features loaded from RAW_DATA.csv")
        return earnings_df
    
    # Merge earnings dates with features
    logger.info("\n3. Combining earnings dates with features...")
    logger.info("-" * 70)
    
    # Merge on company_name, quarter, and year
    combined_df = pd.merge(
        earnings_df,
        features_df,
        on=['company_name', 'quarter', 'year'],
        how='outer',  # Keep all records from both datasets
        suffixes=('_earnings', '_features')
    )
    
    # Consolidate ISIN codes
    if 'isin_code_earnings' in combined_df.columns and 'isin_code_features' in combined_df.columns:
        combined_df['isin_code'] = combined_df['isin_code_earnings'].fillna(combined_df['isin_code_features'])
        combined_df = combined_df.drop(['isin_code_earnings', 'isin_code_features'], axis=1)
    
    logger.info(f"\n✓ Combined dataset: {len(combined_df):,} rows")
    logger.info(f"  Unique companies: {combined_df['company_name'].nunique()}")
    logger.info(f"  Unique features: {combined_df['feature_name'].nunique() if 'feature_name' in combined_df.columns else 0}")
    
    return combined_df
