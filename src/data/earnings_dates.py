"""
Earnings date extraction from quarterly CSV files.

This module extracts earnings announcement dates from the Quarter_*.csv files
and restructures them into a long-format DataFrame suitable for analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

from src.config import Settings


logger = logging.getLogger(__name__)


def extract_company_name(full_name: str) -> str:
    """
    Extract clean company name from the full name string.
    
    The full name typically contains suffixes like "- EARNINGS PER SHARE-REPRT DT-Q1".
    This function extracts just the company name.
    
    Parameters
    ----------
    full_name : str
        Full name string from the CSV (e.g., "AMAZON.COM - EARNINGS PER SHARE-REPRT DT-Q1")
        
    Returns
    -------
    str
        Clean company name (e.g., "AMAZON.COM")
        
    Examples
    --------
    >>> extract_company_name("AMAZON.COM - EARNINGS PER SHARE-REPRT DT-Q1")
    'AMAZON.COM'
    """
    if pd.isna(full_name):
        return ""
    
    # Split on " - " and take the first part
    parts = str(full_name).split(" - ")
    return parts[0].strip() if parts else str(full_name).strip()


def parse_earnings_date(date_str: str, year_col: str) -> pd.Timestamp:
    """
    Parse earnings date from DD.MM.YY format to pandas Timestamp.
    
    Parameters
    ----------
    date_str : str
        Date string in DD.MM.YY format (e.g., "22.04.10")
    year_col : str
        Column name containing the year (e.g., "Q1 2010") to resolve century
        
    Returns
    -------
    pd.Timestamp
        Parsed datetime, or NaT if parsing fails
        
    Examples
    --------
    >>> parse_earnings_date("22.04.10", "Q1 2010")
    Timestamp('2010-04-22 00:00:00')
    """
    if pd.isna(date_str) or str(date_str).strip() == "":
        return pd.NaT
    
    try:
        # Extract year from column name (e.g., "Q1 2010" -> 2010)
        year_match = year_col.split()[-1]
        century_year = int(year_match)
        
        # Parse DD.MM.YY
        parts = str(date_str).strip().split('.')
        if len(parts) != 3:
            return pd.NaT
        
        day = int(parts[0])
        month = int(parts[1])
        year_suffix = int(parts[2])
        
        # Determine full year (use century from column name)
        year = (century_year // 100) * 100 + year_suffix
        
        # Create timestamp
        return pd.Timestamp(year=year, month=month, day=day)
        
    except (ValueError, IndexError, AttributeError):
        return pd.NaT


def load_quarterly_earnings_dates(
    quarter: int,
    settings: Settings,
) -> pd.DataFrame:
    """
    Load earnings dates from a single quarterly file.
    
    Parameters
    ----------
    quarter : int
        Quarter number (1, 2, 3, or 4)
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - company_name: str
        - isin_code: str
        - quarter: int
        - year: int
        - earnings_date: datetime64
        
    Notes
    -----
    The quarterly files contain earnings report dates in wide format
    (one row per company, one column per year). This function converts
    to long format for easier analysis.
    """
    file_path = settings.DATA_DIR / f"Quarter_{quarter}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Quarterly file not found: {file_path}")
    
    logger.info(f"  Loading Quarter {quarter} from {file_path.name}")
    
    # Load the CSV
    df = pd.read_csv(file_path, sep=';', low_memory=False)
    
    logger.info(f"    Raw data: {len(df)} companies × {len(df.columns)} columns")
    
    # Extract company names and ISIN codes
    df['company_name'] = df['#NOM?'].apply(extract_company_name)
    
    # Get year columns (e.g., "Q1 2010", "Q1 2011", ...)
    year_columns = [col for col in df.columns if col.startswith(f'Q{quarter}')]
    
    logger.info(f"    Found {len(year_columns)} year columns")
    
    # Convert to long format
    records = []
    
    for idx, row in df.iterrows():
        company_name = row['company_name']
        isin_code = row.get('ISIN CODE', '')
        
        if pd.isna(company_name) or company_name == '':
            continue
        
        for year_col in year_columns:
            date_str = row[year_col]
            
            # Parse the date
            earnings_date = parse_earnings_date(date_str, year_col)
            
            if pd.notna(earnings_date):
                # Extract year from column name
                year = int(year_col.split()[-1])
                
                records.append({
                    'company_name': company_name,
                    'isin_code': isin_code,
                    'quarter': quarter,
                    'year': year,
                    'earnings_date': earnings_date
                })
    
    result_df = pd.DataFrame(records)
    
    logger.info(f"    ✓ Extracted {len(result_df)} earnings dates")
    
    return result_df


def load_all_earnings_dates(settings: Settings) -> pd.DataFrame:
    """
    Load and combine earnings dates from all quarterly files.
    
    Parameters
    ----------
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all earnings dates from Q1-Q4
        Columns: company_name, isin_code, quarter, year, earnings_date
        
    Notes
    -----
    This function loads all four quarterly files and combines them
    into a single long-format DataFrame.
    """
    logger.info("Loading earnings dates from all quarterly files...")
    
    all_dates = []
    
    for quarter in [1, 2, 3, 4]:
        try:
            quarter_df = load_quarterly_earnings_dates(quarter, settings)
            all_dates.append(quarter_df)
        except FileNotFoundError as e:
            logger.warning(f"  ⚠ {str(e)}")
            continue
        except Exception as e:
            logger.error(f"  ✗ Error loading Quarter {quarter}: {str(e)}")
            continue
    
    if not all_dates:
        raise ValueError("No quarterly earnings data could be loaded")
    
    # Combine all quarters
    combined_df = pd.concat(all_dates, ignore_index=True)
    
    # Sort by company and date
    combined_df = combined_df.sort_values(['company_name', 'earnings_date']).reset_index(drop=True)
    
    logger.info(f"\n✓ Loaded {len(combined_df):,} total earnings dates")
    logger.info(f"  Companies: {combined_df['company_name'].nunique()}")
    logger.info(f"  Date range: {combined_df['earnings_date'].min().date()} to {combined_df['earnings_date'].max().date()}")
    
    return combined_df
