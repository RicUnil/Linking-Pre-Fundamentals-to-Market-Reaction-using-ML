"""
Step 04 — Load SPY / Benchmark CSV Data.

This step loads BENCHMARK.csv from the data directory, performs basic sanity
checks, and saves an intermediate parquet file for the benchmark time series.

The benchmark data typically contains SPY (S&P 500 ETF) total return index
or similar market benchmark information used for calculating excess returns.

Usage
-----
    python -m src.step_04_load_spy_benchmark
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import pandas as pd

from src.config import settings
from src.data.load_data import load_benchmark_data


def run_step_04() -> None:
    """
    Execute Step 04: Load SPY/benchmark data.
    
    This function loads the benchmark dataset, performs basic validation,
    and saves intermediate files for downstream processing.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If BENCHMARK.csv is missing.
    ValueError
        If the benchmark data is empty or invalid.
    Exception
        If any critical step fails.
    """
    # Initialize logging
    logger = settings.setup_logging("step_04_load_spy_benchmark")
    logger.info("=" * 70)
    logger.info("STEP 04: LOAD SPY / BENCHMARK CSV DATA")
    logger.info("=" * 70)
    
    try:
        # Ensure directories exist
        settings.ensure_directories()
        
        # Create step-specific results directory
        step_results_dir = settings.get_step_results_dir(4)
        logger.info(f"\nStep 04 results directory: {step_results_dir}")
        
        # Step 1: Load benchmark dataset
        logger.info("\n" + "-" * 70)
        logger.info("Loading benchmark dataset (BENCHMARK.csv)...")
        logger.info("-" * 70)
        
        benchmark_df = load_benchmark_data(settings)
        
        # Step 2: Basic validation
        logger.info(f"\n✓ Successfully loaded BENCHMARK.csv")
        logger.info(f"  Total rows: {benchmark_df.shape[0]:,}")
        logger.info(f"  Total columns: {benchmark_df.shape[1]}")
        
        # Validate that DataFrame is not empty
        if len(benchmark_df) == 0:
            raise ValueError("Benchmark DataFrame is empty!")
        
        # Check for numeric columns (should have at least one for the index/returns)
        numeric_cols = benchmark_df.select_dtypes(include=['number']).columns.tolist()
        logger.info(f"  Numeric columns: {len(numeric_cols)}")
        if len(numeric_cols) > 0:
            logger.info(f"  Numeric column names: {numeric_cols[:5]}")
        else:
            logger.warning("  ⚠ No numeric columns detected in benchmark data")
        
        # Check for date columns
        date_cols = benchmark_df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols:
            logger.info(f"  Date columns detected: {date_cols}")
            
            # If there's exactly one date column, sort by it
            if len(date_cols) == 1:
                date_col = date_cols[0]
                benchmark_df = benchmark_df.sort_values(by=date_col).reset_index(drop=True)
                logger.info(f"  ✓ Sorted by date column: {date_col}")
                logger.info(f"  Date range: {benchmark_df[date_col].min()} to {benchmark_df[date_col].max()}")
        
        # Display memory usage
        memory_mb = benchmark_df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"  Memory usage: {memory_mb:.2f} MB")
        
        # Step 3: Data quality summary
        logger.info("\n" + "-" * 70)
        logger.info("Basic data quality summary...")
        logger.info("-" * 70)
        
        # Check for missing values
        missing_counts = benchmark_df.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        
        if len(columns_with_missing) > 0:
            logger.info(f"\n  Columns with missing values: {len(columns_with_missing)}/{len(benchmark_df.columns)}")
            for col, count in columns_with_missing.items():
                pct = (count / len(benchmark_df)) * 100
                logger.info(f"    - {col}: {count:,} ({pct:.1f}%)")
        else:
            logger.info(f"\n  ✓ No missing values detected")
        
        # Check for duplicate rows
        n_duplicates = benchmark_df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"  ⚠ Found {n_duplicates:,} duplicate rows")
        else:
            logger.info(f"  ✓ No duplicate rows detected")
        
        # Step 4: Save intermediate files
        logger.info("\n" + "-" * 70)
        logger.info("Saving intermediate files...")
        logger.info("-" * 70)
        
        # Save parquet file
        parquet_path = step_results_dir / "benchmark_raw.parquet"
        benchmark_df.to_parquet(parquet_path, index=False, engine='pyarrow')
        logger.info(f"✓ Saved benchmark data to: {parquet_path}")
        logger.info(f"  File size: {parquet_path.stat().st_size / 1024**2:.2f} MB")
        
        # Save sample CSV
        sample_path = step_results_dir / "benchmark_head.csv"
        benchmark_df.head(10).to_csv(sample_path, index=False)
        logger.info(f"✓ Saved sample (first 10 rows) to: {sample_path}")
        
        # Step 5: Save completion marker
        logger.info("\n" + "-" * 70)
        logger.info("Saving completion marker...")
        logger.info("-" * 70)
        
        completion_file = step_results_dir / "step_04_completed.txt"
        
        completion_message = f"""Step 04 - Load SPY / Benchmark CSV Data
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Benchmark Data Summary:
- File: BENCHMARK.csv
- Rows: {benchmark_df.shape[0]:,}
- Columns: {benchmark_df.shape[1]}
- Numeric columns: {len(numeric_cols)}
- Date columns: {len(date_cols)}

Intermediate Files Created:
- {parquet_path.name} ({parquet_path.stat().st_size / 1024**2:.2f} MB)
- {sample_path.name}

Data Quality Summary:
- Columns with missing values: {len(columns_with_missing)}/{len(benchmark_df.columns)}
- Duplicate rows: {n_duplicates:,}

Status: SUCCESS
Benchmark data successfully loaded and saved for downstream processing.
"""
        
        with open(completion_file, 'w') as f:
            f.write(completion_message)
        
        logger.info(f"✓ Completion marker saved: {completion_file}")
        
        # Final success message
        logger.info("\n" + "=" * 70)
        logger.info("Step 04 completed successfully: benchmark data loaded and saved")
        logger.info("=" * 70)
        logger.info(f"\nKey outputs:")
        logger.info(f"  - {parquet_path}")
        logger.info(f"  - {sample_path}")
        logger.info(f"  - {completion_file}")
        logger.info("\nYou may proceed to Step 05 when ready.")
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ File not found: {str(e)}")
        logger.error("Please ensure BENCHMARK.csv exists in the data directory.")
        raise
    except Exception as e:
        logger.error(f"\n✗ Step 04 failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    run_step_04()
