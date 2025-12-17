"""
Step 03b — Extract earnings announcement dates from quarterly files.

This step:
- Reads the Quarter_1.csv, Quarter_2.csv, Quarter_3.csv, Quarter_4.csv files
- Extracts earnings announcement dates for each company and quarter
- Converts from wide format (one row per company) to long format (one row per event)
- Saves a structured DataFrame with company names, ISIN codes, and earnings dates

This corrects the earnings date extraction to use the actual dates from the quarterly files.

Usage
-----
    python -m src.step_03b_extract_earnings_dates
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data.earnings_dates import load_all_earnings_dates
from src.data.ticker_mapping import add_ticker_column


def run_step_03b() -> None:
    """
    Execute Step 03b: Extract earnings announcement dates.
    
    This function loads the quarterly CSV files and extracts the actual
    earnings announcement dates for each company.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If quarterly CSV files are missing.
    Exception
        If any critical step fails.
    """
    # Initialize logging
    logger = settings.setup_logging("step_03b_extract_earnings_dates")
    logger.info("=" * 70)
    logger.info("STEP 03B: EXTRACT EARNINGS ANNOUNCEMENT DATES")
    logger.info("=" * 70)
    
    try:
        # Ensure directories exist
        settings.ensure_directories()
        
        # Create step-specific results directory
        step_results_dir = settings.get_step_results_dir(3)
        logger.info(f"\nStep 03b results directory: {step_results_dir}")
        
        # Step 1: Load earnings dates from quarterly files
        logger.info("\n" + "-" * 70)
        logger.info("Loading earnings dates from quarterly files...")
        logger.info("-" * 70)
        
        earnings_dates_df = load_all_earnings_dates(settings)
        
        logger.info(f"\n✓ Loaded {len(earnings_dates_df):,} earnings events")
        
        # Step 1b: Add ticker mapping
        logger.info("\n" + "-" * 70)
        logger.info("Mapping company names to ticker symbols...")
        logger.info("-" * 70)
        
        earnings_dates_df = add_ticker_column(earnings_dates_df, 'company_name')
        
        # Filter to only events with ticker mappings
        initial_count = len(earnings_dates_df)
        earnings_dates_df = earnings_dates_df[earnings_dates_df['ticker'].notna()].copy()
        filtered_count = initial_count - len(earnings_dates_df)
        
        if filtered_count > 0:
            logger.info(f"  Filtered out {filtered_count:,} events without ticker mapping")
        
        logger.info(f"  ✓ Retained {len(earnings_dates_df):,} events with ticker symbols")
        
        # Step 2: Data quality summary
        logger.info("\n" + "-" * 70)
        logger.info("Data quality summary...")
        logger.info("-" * 70)
        
        logger.info(f"  Total earnings events: {len(earnings_dates_df):,}")
        logger.info(f"  Unique companies: {earnings_dates_df['company_name'].nunique()}")
        logger.info(f"  Date range: {earnings_dates_df['earnings_date'].min().date()} to {earnings_dates_df['earnings_date'].max().date()}")
        
        # Events per quarter
        quarter_counts = earnings_dates_df['quarter'].value_counts().sort_index()
        logger.info(f"\n  Events per quarter:")
        for q, count in quarter_counts.items():
            logger.info(f"    Q{q}: {count:,} events")
        
        # Events per year
        year_counts = earnings_dates_df['year'].value_counts().sort_index()
        logger.info(f"\n  Events per year (sample):")
        for year in sorted(year_counts.index)[:5]:
            logger.info(f"    {year}: {year_counts[year]:,} events")
        if len(year_counts) > 5:
            logger.info(f"    ... and {len(year_counts) - 5} more years")
        
        # Sample companies
        sample_companies = earnings_dates_df['company_name'].unique()[:10]
        logger.info(f"\n  Sample companies:")
        for company in sample_companies:
            company_events = len(earnings_dates_df[earnings_dates_df['company_name'] == company])
            logger.info(f"    {company}: {company_events} events")
        
        # Step 3: Save structured earnings dates
        logger.info("\n" + "-" * 70)
        logger.info("Saving structured earnings dates...")
        logger.info("-" * 70)
        
        # Save as parquet
        output_path = step_results_dir / "earnings_dates.parquet"
        earnings_dates_df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"✓ Saved earnings dates: {output_path}")
        logger.info(f"  Rows: {len(earnings_dates_df):,}")
        logger.info(f"  Columns: {len(earnings_dates_df.columns)}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")
        
        # Save sample as CSV
        sample_path = step_results_dir / "earnings_dates_head.csv"
        earnings_dates_df.head(50).to_csv(sample_path, index=False)
        logger.info(f"✓ Saved sample (first 50 rows): {sample_path}")
        
        # Step 4: Save completion marker
        logger.info("\n" + "-" * 70)
        logger.info("Saving completion marker...")
        logger.info("-" * 70)
        
        completion_file = step_results_dir / "step_03b_completed.txt"
        
        completion_message = f"""Step 03b - Extract Earnings Announcement Dates
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Data:
- Quarterly files: Quarter_1.csv, Quarter_2.csv, Quarter_3.csv, Quarter_4.csv

Processing Results:
- Total earnings events: {len(earnings_dates_df):,}
- Unique companies: {earnings_dates_df['company_name'].nunique()}
- Date range: {earnings_dates_df['earnings_date'].min().date()} to {earnings_dates_df['earnings_date'].max().date()}

Events per Quarter:
{chr(10).join(f'- Q{q}: {count:,}' for q, count in quarter_counts.items())}

Output Files:
- {output_path.name} ({output_path.stat().st_size / 1024**2:.2f} MB)
- {sample_path.name}

Status: SUCCESS
Earnings announcement dates successfully extracted from quarterly files.
This data can now be used to replace the dummy dates in the pipeline.
"""
        
        with open(completion_file, 'w') as f:
            f.write(completion_message)
        
        logger.info(f"✓ Completion marker saved: {completion_file}")
        
        # Final success message
        logger.info("\n" + "=" * 70)
        logger.info("Step 03b completed successfully: earnings dates extracted")
        logger.info("=" * 70)
        logger.info(f"\nKey outputs:")
        logger.info(f"  - {output_path}")
        logger.info(f"  - {sample_path}")
        logger.info(f"  - {completion_file}")
        logger.info(f"\nTotal events: {len(earnings_dates_df):,}")
        logger.info(f"Companies: {earnings_dates_df['company_name'].nunique()}")
        logger.info("\nYou can now use this data to rerun Steps 06 and 07 with actual earnings dates.")
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"\n✗ Step 03b failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    run_step_03b()
