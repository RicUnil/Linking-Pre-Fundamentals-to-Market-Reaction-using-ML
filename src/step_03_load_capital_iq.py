"""
Step 03 — Load Capital IQ / Earnings CSV Data.

This step loads RAW_DATA.csv and the four quarter files from the data directory,
performs basic sanity checks, and saves an intermediate parquet file for the
main earnings dataset.

The script performs the following tasks:
1. Load the main earnings dataset (RAW_DATA.csv)
2. Load all quarterly supplementary files (Quarter_1.csv through Quarter_4.csv)
3. Log shapes and basic statistics
4. Save intermediate parquet file for future steps
5. Save a sample CSV for quick inspection
6. Create completion marker

Usage
-----
    python -m src.step_03_load_capital_iq
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import pandas as pd

from src.config import settings
from src.data.load_data import (
    load_raw_earnings_data,
    load_all_quarter_files,
)


def run_step_03() -> None:
    """
    Execute Step 03: Load Capital IQ earnings data.
    
    This function loads the main earnings dataset and quarterly files,
    performs basic validation, and saves intermediate files for downstream
    processing.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If any required data file is missing.
    Exception
        If any critical step fails.
    """
    # Initialize logging
    logger = settings.setup_logging("step_03_load_capital_iq")
    logger.info("=" * 70)
    logger.info("STEP 03: LOAD CAPITAL IQ / EARNINGS CSV DATA")
    logger.info("=" * 70)
    
    try:
        # Ensure directories exist
        settings.ensure_directories()
        
        # Create step-specific results directory
        step_results_dir = settings.get_step_results_dir(3)
        logger.info(f"\nStep 03 results directory: {step_results_dir}")
        
        # Step 1: Load main earnings dataset
        logger.info("\n" + "-" * 70)
        logger.info("Loading main earnings dataset (RAW_DATA.csv)...")
        logger.info("-" * 70)
        
        earnings_df = load_raw_earnings_data(settings)
        
        logger.info(f"\n✓ Successfully loaded RAW_DATA.csv")
        logger.info(f"  Total rows: {earnings_df.shape[0]:,}")
        logger.info(f"  Total columns: {earnings_df.shape[1]}")
        logger.info(f"  Memory usage: {earnings_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display column names (first 15)
        logger.info(f"\n  Column names (first 15):")
        for i, col in enumerate(earnings_df.columns[:15], 1):
            logger.info(f"    {i:2d}. {col}")
        if len(earnings_df.columns) > 15:
            logger.info(f"    ... and {len(earnings_df.columns) - 15} more columns")
        
        # Step 2: Load quarterly files
        logger.info("\n" + "-" * 70)
        logger.info("Loading quarterly supplementary files...")
        logger.info("-" * 70)
        
        q1_df, q2_df, q3_df, q4_df = load_all_quarter_files(settings)
        
        # Log summary of quarterly files
        logger.info(f"\n✓ Successfully loaded all quarterly files")
        logger.info(f"  Quarter 1: {q1_df.shape[0]:,} rows × {q1_df.shape[1]} columns")
        logger.info(f"  Quarter 2: {q2_df.shape[0]:,} rows × {q2_df.shape[1]} columns")
        logger.info(f"  Quarter 3: {q3_df.shape[0]:,} rows × {q3_df.shape[1]} columns")
        logger.info(f"  Quarter 4: {q4_df.shape[0]:,} rows × {q4_df.shape[1]} columns")
        
        # Step 3: Save intermediate parquet file
        logger.info("\n" + "-" * 70)
        logger.info("Saving intermediate files...")
        logger.info("-" * 70)
        
        parquet_path = step_results_dir / "earnings_raw.parquet"
        earnings_df.to_parquet(parquet_path, index=False, engine='pyarrow')
        logger.info(f"✓ Saved earnings data to: {parquet_path}")
        logger.info(f"  File size: {parquet_path.stat().st_size / 1024**2:.2f} MB")
        
        # Step 4: Save a sample CSV for quick inspection
        sample_path = step_results_dir / "earnings_head.csv"
        earnings_df.head(10).to_csv(sample_path, index=False)
        logger.info(f"✓ Saved sample (first 10 rows) to: {sample_path}")
        
        # Step 5: Basic data quality checks
        logger.info("\n" + "-" * 70)
        logger.info("Basic data quality summary...")
        logger.info("-" * 70)
        
        # Check for missing values
        missing_counts = earnings_df.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        
        if len(columns_with_missing) > 0:
            logger.info(f"\n  Columns with missing values: {len(columns_with_missing)}/{len(earnings_df.columns)}")
            logger.info(f"  Top 5 columns by missing count:")
            for col, count in columns_with_missing.nlargest(5).items():
                pct = (count / len(earnings_df)) * 100
                logger.info(f"    - {col}: {count:,} ({pct:.1f}%)")
        else:
            logger.info(f"\n  ✓ No missing values detected")
        
        # Check for duplicate rows
        n_duplicates = earnings_df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"  ⚠ Found {n_duplicates:,} duplicate rows")
        else:
            logger.info(f"  ✓ No duplicate rows detected")
        
        # Step 6: Save completion marker
        logger.info("\n" + "-" * 70)
        logger.info("Saving completion marker...")
        logger.info("-" * 70)
        
        completion_file = step_results_dir / "step_03_completed.txt"
        
        completion_message = f"""Step 03 - Load Capital IQ / Earnings CSV Data
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Files Loaded:
- RAW_DATA.csv: {earnings_df.shape[0]:,} rows × {earnings_df.shape[1]} columns
- Quarter_1.csv: {q1_df.shape[0]:,} rows × {q1_df.shape[1]} columns
- Quarter_2.csv: {q2_df.shape[0]:,} rows × {q2_df.shape[1]} columns
- Quarter_3.csv: {q3_df.shape[0]:,} rows × {q3_df.shape[1]} columns
- Quarter_4.csv: {q4_df.shape[0]:,} rows × {q4_df.shape[1]} columns

Intermediate Files Created:
- {parquet_path.name} ({parquet_path.stat().st_size / 1024**2:.2f} MB)
- {sample_path.name}

Data Quality Summary:
- Columns with missing values: {len(columns_with_missing)}/{len(earnings_df.columns)}
- Duplicate rows: {n_duplicates:,}

Status: SUCCESS
Earnings data successfully loaded and saved for downstream processing.
"""
        
        with open(completion_file, 'w') as f:
            f.write(completion_message)
        
        logger.info(f"✓ Completion marker saved: {completion_file}")
        
        # Final success message
        logger.info("\n" + "=" * 70)
        logger.info("Step 03 completed successfully: earnings data loaded and saved")
        logger.info("=" * 70)
        logger.info(f"\nKey outputs:")
        logger.info(f"  - {parquet_path}")
        logger.info(f"  - {sample_path}")
        logger.info(f"  - {completion_file}")
        logger.info("\nYou may proceed to Step 04 when ready.")
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ File not found: {str(e)}")
        logger.error("Please ensure all required CSV files exist in the data directory.")
        raise
    except Exception as e:
        logger.error(f"\n✗ Step 03 failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    run_step_03()
