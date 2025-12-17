"""
Clean and consolidate comprehensive dataset.

This module:
- Consolidates duplicate company names
- Pivots features from long to wide format
- Merges earnings dates with features
- Creates a final clean dataset ready for analysis
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd
import numpy as np

from src.config import Settings
from src.data.ticker_mapping import COMPANY_TO_TICKER


logger = logging.getLogger(__name__)


# Company name consolidation mapping
COMPANY_NAME_CONSOLIDATION = {
    "3M": "3M COMPANY",
    "ABBVIE": "ABBVIE INC",
    "ACCENTURE CLASS A": "ACCENTURE PLC",
    "ADOBE (NAS)": "ADOBE",
    "ADVANCED MICRO": "ADVANCED MICRO DEVICES",
    "AES": "AES CORP",
    "AFLAC": "AFLAC INC",
    "AGILENT TECHS.": "AGILENT TECHNOLOGIES",
    "AIR PRDS.& CHEMS.": "AIR PRODUCTS & CHEMICALS",
    "AIRBNB A": "AIRBNB",
    "ALBEMARLE": "ALBEMARLE CORP",
    "ALIGN TECHNOLOGY": "ALIGN TECHNOLOGY INC",
    "ALTRIA": "ALTRIA GROUP",
    "AMAZON.COM": "AMAZON",
    "AMERICAN TOWER": "AMERICAN TOWER CORP",
    "ARCHER-DANIELS-MIDLAND": "ARCHER DANIELS MIDLAND",
    "ATLASSIAN CORP": "ATLASSIAN",
    "AVALONBAY COMMUNITIES": "AVALONBAY",
    "BANK OF AMERICA": "BANK OF AMERICA CORP",
    "BOSTON PROPERTIES": "BOSTON PROPERTIES INC",
    "CADENCE DESIGN SYSTEMS": "CADENCE DESIGN",
    "CAPITAL ONE FINANCIAL": "CAPITAL ONE",
    "CARNIVAL CORP": "CARNIVAL",
    "CHARTER COMMUNICATIONS": "CHARTER COMM",
    "CHIPOTLE MEXICAN GRILL": "CHIPOTLE",
    "COGNIZANT TECH.SOLUTIONS": "COGNIZANT",
    "COSTCO WHOLESALE": "COSTCO",
    "CROWDSTRIKE HOLDINGS": "CROWDSTRIKE",
    "DIGITAL REALTY TRUST": "DIGITAL REALTY",
    "DISNEY (WALT)": "WALT DISNEY",
    "DOLLAR GENERAL": "DOLLAR GENERAL CORP",
    "DOLLAR TREE": "DOLLAR TREE INC",
    "DOMINO'S PIZZA": "DOMINOS PIZZA",
    "ESTEE LAUDER": "ESTEE LAUDER COMPANIES",
    "FACEBOOK": "META PLATFORMS",
    "META PLATFORMS INC": "META PLATFORMS",
    "GENERAL ELECTRIC": "GENERAL ELECTRIC CO",
    "GENERAL MOTORS": "GENERAL MOTORS CO",
    "GOLDMAN SACHS GROUP": "GOLDMAN SACHS",
    "GOOGLE": "ALPHABET INC",
    "ALPHABET INC": "ALPHABET",
    "HOME DEPOT": "HOME DEPOT INC",
    "HONEYWELL INTERNATIONAL": "HONEYWELL",
    "INT'L BUSINESS MACHS": "INTERNATIONAL BUSINESS MACHINES",
    "INTERNATIONAL BUS.MCHS.": "INTERNATIONAL BUSINESS MACHINES",
    "JOHNSON & JOHNSON": "JOHNSON AND JOHNSON",
    "JPMORGAN CHASE": "JPMORGAN CHASE & CO",
    "LINDE PLC": "LINDE",
    "LOWE'S": "LOWES",
    "MCDONALD'S": "MCDONALDS",
    "MONDELEZ INTERNATIONAL": "MONDELEZ",
    "MORGAN STANLEY": "MORGAN STANLEY DEAN WITTER",
    "PAYPAL HOLDINGS": "PAYPAL",
    "PHILIP MORRIS INTERNATIONAL": "PHILIP MORRIS INTL",
    "PROCTER & GAMBLE": "PROCTER AND GAMBLE",
    "REGENERON PHARMACEUTICALS": "REGENERON",
    "ROSS STORES": "ROSS STORES INC",
    "SALESFORCE.COM": "SALESFORCE",
    "SCHLUMBERGER": "SCHLUMBERGER LTD",
    "STARBUCKS": "STARBUCKS CORP",
    "SYNOPSYS": "SYNOPSYS INC",
    "TARGET": "TARGET CORP",
    "TESLA": "TESLA INC",
    "TEXAS INSTRUMENTS": "TEXAS INSTRUMENTS INC",
    "THERMO FISHER SCIENTIFIC": "THERMO FISHER",
    "T-MOBILE US": "T-MOBILE",
    "UBER TECHNOLOGIES": "UBER",
    "UNION PACIFIC": "UNION PACIFIC CORP",
    "UNITED PARCEL SERVICE": "UPS",
    "UNITEDHEALTH GROUP": "UNITEDHEALTH",
    "VERTEX PHARMACEUTICALS": "VERTEX",
    "WALMART": "WALMART INC",
    "WALT DISNEY": "DISNEY",
    "WELLS FARGO": "WELLS FARGO & CO",
    "ZOETIS": "ZOETIS INC",
}


def consolidate_company_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate duplicate company names.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with company_name column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with consolidated company names
    """
    df = df.copy()
    
    # Apply consolidation mapping
    df['company_name'] = df['company_name'].replace(COMPANY_NAME_CONSOLIDATION)
    
    # Also standardize to uppercase and strip whitespace
    df['company_name'] = df['company_name'].str.upper().str.strip()
    
    logger.info(f"  Consolidated to {df['company_name'].nunique()} unique companies")
    
    return df


def pivot_features_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot features from long format to wide format.
    
    Each feature becomes a separate column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with feature_name and value columns
        
    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with one column per feature
    """
    logger.info("  Pivoting features to wide format...")
    
    # Filter to rows with features
    features_df = df[df['feature_name'].notna()].copy()
    
    if features_df.empty:
        logger.warning("  No features to pivot")
        return df
    
    # Pivot: rows are (company_name, quarter, year), columns are features
    pivoted = features_df.pivot_table(
        index=['company_name', 'isin_code', 'quarter', 'year'],
        columns='feature_name',
        values='value',
        aggfunc='first'  # Take first value if duplicates
    ).reset_index()
    
    # Clean column names (remove spaces, special chars)
    pivoted.columns.name = None  # Remove the 'feature_name' label
    
    # Rename columns to be more Python-friendly
    column_mapping = {}
    for col in pivoted.columns:
        if col not in ['company_name', 'isin_code', 'quarter', 'year']:
            # Convert to snake_case
            clean_col = str(col).lower().replace(' ', '_').replace('/', '_').replace("'", '').replace('.', '')
            clean_col = clean_col.replace('-', '_').replace('&', 'and')
            column_mapping[col] = clean_col
    
    pivoted = pivoted.rename(columns=column_mapping)
    
    logger.info(f"  ✓ Pivoted to {len(pivoted)} rows × {len(pivoted.columns)} columns")
    
    return pivoted


def merge_earnings_and_features(
    earnings_df: pd.DataFrame,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge earnings dates with features.
    
    Parameters
    ----------
    earnings_df : pd.DataFrame
        DataFrame with earnings dates
    features_df : pd.DataFrame
        DataFrame with features (wide format)
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame
    """
    logger.info("  Merging earnings dates with features...")
    
    # Merge on company_name, quarter, year
    merged = pd.merge(
        earnings_df,
        features_df,
        on=['company_name', 'quarter', 'year'],
        how='outer',
        suffixes=('', '_features')
    )
    
    # Consolidate ISIN codes if there are duplicates
    if 'isin_code_features' in merged.columns:
        merged['isin_code'] = merged['isin_code'].fillna(merged['isin_code_features'])
        merged = merged.drop('isin_code_features', axis=1)
    
    logger.info(f"  ✓ Merged to {len(merged)} rows")
    
    return merged


def clean_and_consolidate_data(settings: Settings) -> pd.DataFrame:
    """
    Clean and consolidate the comprehensive dataset.
    
    Parameters
    ----------
    settings : Settings
        Global project settings
        
    Returns
    -------
    pd.DataFrame
        Clean, consolidated dataset ready for analysis
    """
    logger.info("=" * 70)
    logger.info("CLEANING AND CONSOLIDATING DATA")
    logger.info("=" * 70)
    
    # Load comprehensive data
    logger.info("\n1. Loading comprehensive data...")
    logger.info("-" * 70)
    
    comp_data_path = settings.get_step_results_dir(3) / "comprehensive_data.parquet"
    
    if not comp_data_path.exists():
        raise FileNotFoundError(f"Comprehensive data not found: {comp_data_path}")
    
    df = pd.read_parquet(comp_data_path)
    logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Consolidate company names
    logger.info("\n2. Consolidating company names...")
    logger.info("-" * 70)
    
    df = consolidate_company_names(df)
    
    # Separate earnings dates and features
    logger.info("\n3. Separating earnings dates and features...")
    logger.info("-" * 70)
    
    earnings_df = df[df['earnings_date'].notna()][['company_name', 'isin_code', 'quarter', 'year', 'earnings_date']].drop_duplicates()
    logger.info(f"  Earnings dates: {len(earnings_df):,} rows")
    
    # Pivot features to wide format
    logger.info("\n4. Pivoting features to wide format...")
    logger.info("-" * 70)
    
    features_wide = pivot_features_to_wide(df)
    
    # Merge earnings and features
    logger.info("\n5. Merging earnings dates with features...")
    logger.info("-" * 70)
    
    final_df = merge_earnings_and_features(earnings_df, features_wide)
    
    # Add ticker mapping
    logger.info("\n6. Adding ticker symbols...")
    logger.info("-" * 70)
    
    # Create ticker mapping
    ticker_map = {}
    for company in final_df['company_name'].unique():
        # Try to find ticker
        ticker = None
        for key, val in COMPANY_TO_TICKER.items():
            if key.upper() == company.upper():
                ticker = val
                break
        if ticker:
            ticker_map[company] = ticker
    
    final_df['ticker'] = final_df['company_name'].map(ticker_map)
    
    mapped_count = final_df['ticker'].notna().sum()
    logger.info(f"  Mapped {mapped_count:,} / {len(final_df):,} rows to tickers ({100*mapped_count/len(final_df):.1f}%)")
    
    # Sort by company, year, quarter
    final_df = final_df.sort_values(['company_name', 'year', 'quarter']).reset_index(drop=True)
    
    logger.info(f"\n✓ Final dataset: {len(final_df):,} rows × {len(final_df.columns)} columns")
    logger.info(f"  Companies: {final_df['company_name'].nunique()}")
    logger.info(f"  With tickers: {final_df['ticker'].nunique()}")
    logger.info(f"  With earnings dates: {final_df['earnings_date'].notna().sum():,}")
    
    return final_df
