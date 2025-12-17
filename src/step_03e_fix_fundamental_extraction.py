"""
Step 03e — Fix fundamental data extraction.

This step properly extracts fundamental data from RAW_DATA.csv by:
1. Parsing company names from feature descriptions
2. Pivoting quarters from columns to rows
3. Pivoting features from rows to columns
4. Merging with earnings dates
5. Adding ticker symbols

This fixes the data alignment issue that prevented ratio calculations.
"""

from typing import NoReturn
import logging
from pathlib import Path

import pandas as pd

from src.config import Settings
from src.data.extract_fundamentals import (
    extract_fundamentals_from_raw_data,
    merge_with_earnings_dates,
    add_ticker_mapping,
)


def run_step_03e() -> NoReturn:
    """
    Execute Step 03e: Fix fundamental data extraction.
    
    This step creates a properly structured dataset where all fundamental
    features are aligned in the same rows, enabling ratio calculations.
    """
    # Initialize settings
    settings = Settings()
    logger = settings.setup_logging("step_03e_fix_fundamental_extraction")
    
    logger.info("=" * 70)
    logger.info("STEP 03e: FIX FUNDAMENTAL DATA EXTRACTION")
    logger.info("=" * 70)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Create step 03 results directory
    step_03_dir = settings.get_step_results_dir(3)
    step_03_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nStep 03e results directory: {step_03_dir}")
    
    # ========================================================================
    # Extract fundamental data from RAW_DATA.csv
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Extracting fundamental data from RAW_DATA.csv...")
    logger.info("-" * 70)
    
    raw_data_path = settings.DATA_DIR / "RAW_DATA.csv"
    
    if not raw_data_path.exists():
        error_msg = f"RAW_DATA.csv not found at {raw_data_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    fundamentals_df = extract_fundamentals_from_raw_data(raw_data_path, settings)
    
    logger.info(f"\n✓ Extracted fundamental data:")
    logger.info(f"  Rows: {len(fundamentals_df):,}")
    logger.info(f"  Columns: {len(fundamentals_df.columns)}")
    logger.info(f"  Companies: {fundamentals_df['company_name'].nunique()}")
    logger.info(f"  Quarters: {fundamentals_df['quarter'].nunique()}")
    logger.info(f"  Years: {fundamentals_df['year'].min()}-{fundamentals_df['year'].max()}")
    
    # ========================================================================
    # Load earnings dates
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Loading earnings dates...")
    logger.info("-" * 70)
    
    # Check if earnings_dates.parquet exists (from step 03b)
    earnings_path = step_03_dir / "earnings_dates.parquet"
    
    if earnings_path.exists():
        logger.info(f"  Loading from {earnings_path}")
        earnings_df = pd.read_parquet(earnings_path)
    else:
        # Fall back to loading from Quarter files
        logger.info("  earnings_dates.parquet not found, loading from Quarter files...")
        from src.data.earnings_dates import load_and_restructure_earnings_dates
        
        earnings_df = load_and_restructure_earnings_dates(settings)
    
    logger.info(f"  Loaded {len(earnings_df):,} earnings events")
    
    # ========================================================================
    # Merge with earnings dates
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Merging with earnings dates...")
    logger.info("-" * 70)
    
    merged_df = merge_with_earnings_dates(fundamentals_df, earnings_df, settings)
    
    # ========================================================================
    # Add ticker symbols
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Adding ticker symbols...")
    logger.info("-" * 70)
    
    final_df = add_ticker_mapping(merged_df, settings)
    
    # ========================================================================
    # Data quality summary
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Data quality summary...")
    logger.info("-" * 70)
    
    # Count non-null, non-zero values for key features
    feature_cols = [col for col in final_df.columns 
                   if col not in ['company_name', 'quarter', 'year', 'earnings_date', 'ticker']]
    
    logger.info(f"\n  Feature columns: {len(feature_cols)}")
    logger.info(f"\n  Top 10 features by data availability:")
    
    feature_coverage = []
    for col in feature_cols:
        non_null = final_df[col].notna().sum()
        non_zero = ((final_df[col] != 0) & final_df[col].notna()).sum()
        if non_zero > 0:
            feature_coverage.append((col, non_zero, 100 * non_zero / len(final_df)))
    
    feature_coverage.sort(key=lambda x: x[1], reverse=True)
    
    for col, count, pct in feature_coverage[:10]:
        logger.info(f"    {col:40s}: {count:6,} ({pct:5.1f}%)")
    
    # ========================================================================
    # Save fixed data
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving fixed fundamental data...")
    logger.info("-" * 70)
    
    # Save as clean_data_fixed.parquet (don't overwrite existing clean_data.parquet)
    output_path = step_03_dir / "clean_data_fixed.parquet"
    final_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    
    file_size_mb = output_path.stat().st_size / (1024**2)
    
    logger.info(f"✓ Saved fixed data: {output_path}")
    logger.info(f"  Rows: {len(final_df):,}")
    logger.info(f"  Columns: {len(final_df.columns)}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    
    # Save sample
    sample_path = step_03_dir / "clean_data_fixed_head.csv"
    final_df.head(50).to_csv(sample_path, index=False)
    
    logger.info(f"✓ Saved sample (first 50 rows): {sample_path}")
    
    # ========================================================================
    # Save completion marker
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving completion marker...")
    logger.info("-" * 70)
    
    completion_path = step_03_dir / "step_03e_completed.txt"
    
    with open(completion_path, 'w') as f:
        f.write("Step 03e: Fix Fundamental Data Extraction\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Status: COMPLETED\n\n")
        f.write(f"Input: {raw_data_path}\n")
        f.write(f"Output: {output_path}\n\n")
        f.write(f"Rows: {len(final_df):,}\n")
        f.write(f"Columns: {len(final_df.columns)}\n")
        f.write(f"Companies: {final_df['company_name'].nunique()}\n")
        f.write(f"With earnings dates: {final_df['earnings_date'].notna().sum():,}\n")
        f.write(f"With tickers: {final_df['ticker'].notna().sum():,}\n\n")
        f.write(f"Feature columns: {len(feature_cols)}\n")
        f.write(f"Features with data: {len(feature_coverage)}\n\n")
        f.write("Top features by coverage:\n")
        for col, count, pct in feature_coverage[:15]:
            f.write(f"  - {col}: {count:,} ({pct:.1f}%)\n")
    
    logger.info(f"✓ Completion marker saved: {completion_path}")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 03e completed successfully: fundamental data properly extracted")
    logger.info("=" * 70)
    logger.info("\nKey outputs:")
    logger.info(f"  - {output_path}")
    logger.info(f"  - {sample_path}")
    logger.info(f"  - {completion_path}")
    logger.info(f"\nData summary:")
    logger.info(f"  Total rows: {len(final_df):,}")
    logger.info(f"  Feature columns: {len(feature_cols)}")
    logger.info(f"  Features with data: {len(feature_coverage)}")
    logger.info(f"  Companies: {final_df['company_name'].nunique()}")
    logger.info(f"  With tickers: {final_df['ticker'].notna().sum():,}")
    logger.info("\nNext steps:")
    logger.info("  1. Update Step 07 to use clean_data_fixed.parquet")
    logger.info("  2. Re-run Step 08 to compute features with full data")
    logger.info("  3. Proceed to Step 09 for market features")


if __name__ == "__main__":
    run_step_03e()
