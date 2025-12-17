"""
Step 03d — Clean and consolidate comprehensive dataset.

This step:
- Consolidates duplicate company names
- Pivots features from long to wide format
- Merges earnings dates with features
- Adds ticker symbols
- Creates final clean dataset ready for analysis

Usage
-----
    python -m src.step_03d_clean_and_consolidate
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data.clean_and_consolidate import clean_and_consolidate_data


def run_step_03d() -> None:
    """
    Execute Step 03d: Clean and consolidate comprehensive dataset.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If comprehensive data file is missing.
    Exception
        If any critical step fails.
    """
    # Initialize logging
    logger = settings.setup_logging("step_03d_clean_and_consolidate")
    logger.info("=" * 70)
    logger.info("STEP 03D: CLEAN AND CONSOLIDATE COMPREHENSIVE DATASET")
    logger.info("=" * 70)
    
    try:
        # Ensure directories exist
        settings.ensure_directories()
        
        # Create step-specific results directory
        step_results_dir = settings.get_step_results_dir(3)
        logger.info(f"\nStep 03d results directory: {step_results_dir}")
        
        # Clean and consolidate data
        final_df = clean_and_consolidate_data(settings)
        
        # Data quality summary
        logger.info("\n" + "=" * 70)
        logger.info("FINAL DATASET SUMMARY")
        logger.info("=" * 70)
        
        logger.info(f"\nDimensions:")
        logger.info(f"  Total rows: {len(final_df):,}")
        logger.info(f"  Total columns: {len(final_df.columns)}")
        
        logger.info(f"\nCompanies:")
        logger.info(f"  Unique companies: {final_df['company_name'].nunique()}")
        logger.info(f"  With ticker symbols: {final_df['ticker'].nunique()}")
        logger.info(f"  With earnings dates: {final_df[final_df['earnings_date'].notna()]['company_name'].nunique()}")
        
        logger.info(f"\nData Coverage:")
        logger.info(f"  Rows with earnings dates: {final_df['earnings_date'].notna().sum():,} ({100*final_df['earnings_date'].notna().sum()/len(final_df):.1f}%)")
        logger.info(f"  Rows with ticker symbols: {final_df['ticker'].notna().sum():,} ({100*final_df['ticker'].notna().sum()/len(final_df):.1f}%)")
        
        if 'earnings_date' in final_df.columns:
            earnings_dates = final_df['earnings_date'].dropna()
            if len(earnings_dates) > 0:
                logger.info(f"  Earnings date range: {earnings_dates.min().date()} to {earnings_dates.max().date()}")
        
        logger.info(f"\nTime Period:")
        logger.info(f"  Years: {final_df['year'].min()} to {final_df['year'].max()}")
        logger.info(f"  Quarters: Q1, Q2, Q3, Q4")
        
        # Feature columns
        feature_cols = [col for col in final_df.columns if col not in ['company_name', 'isin_code', 'quarter', 'year', 'earnings_date', 'ticker', 'company_id']]
        logger.info(f"\nFeature Columns ({len(feature_cols)}):")
        for col in feature_cols:
            non_null = final_df[col].notna().sum()
            logger.info(f"  {col:40s} - {non_null:>6,} non-null ({100*non_null/len(final_df):>5.1f}%)")
        
        # Save final dataset
        logger.info("\n" + "-" * 70)
        logger.info("Saving final clean dataset...")
        logger.info("-" * 70)
        
        # Save as parquet
        output_path = step_results_dir / "clean_data.parquet"
        final_df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"✓ Saved clean dataset: {output_path}")
        logger.info(f"  Rows: {len(final_df):,}")
        logger.info(f"  Columns: {len(final_df.columns)}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")
        
        # Save sample as CSV
        sample_path = step_results_dir / "clean_data_head.csv"
        final_df.head(100).to_csv(sample_path, index=False)
        logger.info(f"✓ Saved sample (first 100 rows): {sample_path}")
        
        # Save column info
        column_info_path = step_results_dir / "clean_data_columns.txt"
        with open(column_info_path, 'w') as f:
            f.write("Clean Dataset Columns\n")
            f.write("=" * 70 + "\n\n")
            for col in final_df.columns:
                dtype = final_df[col].dtype
                non_null = final_df[col].notna().sum()
                null_pct = 100 * (1 - non_null / len(final_df))
                f.write(f"{col:40s} {str(dtype):15s} {non_null:>10,} non-null ({null_pct:5.1f}% null)\n")
        logger.info(f"✓ Saved column info: {column_info_path}")
        
        # Save companies with tickers
        ticker_companies_path = step_results_dir / "companies_with_tickers.csv"
        ticker_companies = final_df[final_df['ticker'].notna()][['company_name', 'ticker', 'isin_code']].drop_duplicates().sort_values('company_name')
        ticker_companies.to_csv(ticker_companies_path, index=False)
        logger.info(f"✓ Saved companies with tickers: {ticker_companies_path}")
        logger.info(f"  {len(ticker_companies)} companies with ticker symbols")
        
        # Save completion marker
        logger.info("\n" + "-" * 70)
        logger.info("Saving completion marker...")
        logger.info("-" * 70)
        
        completion_file = step_results_dir / "step_03d_completed.txt"
        
        completion_message = f"""Step 03d - Clean and Consolidate Comprehensive Dataset
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Data:
- comprehensive_data.parquet (46,369 rows)

Processing Steps:
1. Consolidated duplicate company names
2. Pivoted features from long to wide format
3. Merged earnings dates with features
4. Added ticker symbols

Final Dataset:
- Total rows: {len(final_df):,}
- Total columns: {len(final_df.columns)}
- Unique companies: {final_df['company_name'].nunique()}
- Companies with tickers: {final_df['ticker'].nunique()}
- Rows with earnings dates: {final_df['earnings_date'].notna().sum():,} ({100*final_df['earnings_date'].notna().sum()/len(final_df):.1f}%)
- Feature columns: {len(feature_cols)}

Output Files:
- {output_path.name} ({output_path.stat().st_size / 1024**2:.2f} MB)
- {sample_path.name}
- {column_info_path.name}
- {ticker_companies_path.name}

Status: SUCCESS
Clean dataset ready for pipeline integration.
All companies, features, and earnings dates consolidated.
"""
        
        with open(completion_file, 'w') as f:
            f.write(completion_message)
        
        logger.info(f"✓ Completion marker saved: {completion_file}")
        
        # Final success message
        logger.info("\n" + "=" * 70)
        logger.info("Step 03d completed successfully: data cleaned and consolidated")
        logger.info("=" * 70)
        logger.info(f"\nKey outputs:")
        logger.info(f"  - {output_path}")
        logger.info(f"  - {sample_path}")
        logger.info(f"  - {ticker_companies_path}")
        logger.info(f"  - {completion_file}")
        logger.info(f"\nFinal dataset: {len(final_df):,} rows × {len(final_df.columns)} columns")
        logger.info(f"Companies: {final_df['company_name'].nunique()} ({final_df['ticker'].nunique()} with tickers)")
        logger.info("\n✓ Data is ready for pipeline integration!")
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"\n✗ Step 03d failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    run_step_03d()
