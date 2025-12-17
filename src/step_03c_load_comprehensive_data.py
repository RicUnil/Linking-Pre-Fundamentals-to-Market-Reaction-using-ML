"""
Step 03c — Load and combine ALL data from all CSV files.

This step:
- Loads earnings dates from Quarter_1.csv, Quarter_2.csv, Quarter_3.csv, Quarter_4.csv
- Loads all financial features from RAW_DATA.csv
- Combines everything into a comprehensive dataset
- Preserves N/A values (missing data is intentional)
- Saves the complete dataset for further processing

Usage
-----
    python -m src.step_03c_load_comprehensive_data
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data.comprehensive_data_loader import combine_all_data


def run_step_03c() -> None:
    """
    Execute Step 03c: Load comprehensive dataset from all CSV files.
    
    This function loads and combines all data from the quarterly files
    and RAW_DATA.csv, creating a complete dataset with all companies,
    all features, and all quarters.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If required CSV files are missing.
    Exception
        If any critical step fails.
    """
    # Initialize logging
    logger = settings.setup_logging("step_03c_load_comprehensive_data")
    logger.info("=" * 70)
    logger.info("STEP 03C: LOAD COMPREHENSIVE DATA FROM ALL CSV FILES")
    logger.info("=" * 70)
    
    try:
        # Ensure directories exist
        settings.ensure_directories()
        
        # Create step-specific results directory
        step_results_dir = settings.get_step_results_dir(3)
        logger.info(f"\nStep 03c results directory: {step_results_dir}")
        
        # Load and combine all data
        logger.info("\n" + "-" * 70)
        logger.info("Loading and combining all data...")
        logger.info("-" * 70)
        
        combined_df = combine_all_data(settings)
        
        if combined_df.empty:
            raise ValueError("No data was loaded. Please check CSV files.")
        
        logger.info(f"\n✓ Loaded comprehensive dataset: {len(combined_df):,} rows")
        
        # Data quality summary
        logger.info("\n" + "-" * 70)
        logger.info("Data quality summary...")
        logger.info("-" * 70)
        
        logger.info(f"  Total rows: {len(combined_df):,}")
        logger.info(f"  Total columns: {len(combined_df.columns)}")
        logger.info(f"  Unique companies: {combined_df['company_name'].nunique()}")
        
        if 'earnings_date' in combined_df.columns:
            earnings_count = combined_df['earnings_date'].notna().sum()
            logger.info(f"  Rows with earnings dates: {earnings_count:,} ({100*earnings_count/len(combined_df):.1f}%)")
            
            if earnings_count > 0:
                logger.info(f"  Earnings date range: {combined_df['earnings_date'].min().date()} to {combined_df['earnings_date'].max().date()}")
        
        if 'feature_name' in combined_df.columns:
            unique_features = combined_df['feature_name'].nunique()
            logger.info(f"  Unique features: {unique_features}")
            
            # Show sample features
            sample_features = combined_df['feature_name'].dropna().unique()[:10]
            logger.info(f"  Sample features:")
            for feat in sample_features:
                logger.info(f"    - {feat}")
        
        if 'value' in combined_df.columns:
            non_null_values = combined_df['value'].notna().sum()
            logger.info(f"  Non-null feature values: {non_null_values:,} ({100*non_null_values/len(combined_df):.1f}%)")
        
        # Year and quarter distribution
        if 'year' in combined_df.columns:
            year_counts = combined_df['year'].value_counts().sort_index()
            logger.info(f"\n  Year distribution (sample):")
            for year in sorted(year_counts.index)[:5]:
                logger.info(f"    {year}: {year_counts[year]:,} rows")
            if len(year_counts) > 5:
                logger.info(f"    ... and {len(year_counts) - 5} more years")
        
        if 'quarter' in combined_df.columns:
            quarter_counts = combined_df['quarter'].value_counts().sort_index()
            logger.info(f"\n  Quarter distribution:")
            for q in sorted(quarter_counts.index):
                logger.info(f"    Q{q}: {quarter_counts[q]:,} rows")
        
        # Sample companies
        sample_companies = combined_df['company_name'].unique()[:10]
        logger.info(f"\n  Sample companies:")
        for company in sample_companies:
            company_rows = len(combined_df[combined_df['company_name'] == company])
            logger.info(f"    {company}: {company_rows} rows")
        
        # Save comprehensive dataset
        logger.info("\n" + "-" * 70)
        logger.info("Saving comprehensive dataset...")
        logger.info("-" * 70)
        
        # Save as parquet (efficient for large datasets)
        output_path = step_results_dir / "comprehensive_data.parquet"
        combined_df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"✓ Saved comprehensive dataset: {output_path}")
        logger.info(f"  Rows: {len(combined_df):,}")
        logger.info(f"  Columns: {len(combined_df.columns)}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")
        
        # Save sample as CSV for inspection
        sample_path = step_results_dir / "comprehensive_data_head.csv"
        combined_df.head(100).to_csv(sample_path, index=False)
        logger.info(f"✓ Saved sample (first 100 rows): {sample_path}")
        
        # Save column info
        column_info_path = step_results_dir / "comprehensive_data_columns.txt"
        with open(column_info_path, 'w') as f:
            f.write("Comprehensive Dataset Columns\n")
            f.write("=" * 70 + "\n\n")
            for col in combined_df.columns:
                dtype = combined_df[col].dtype
                non_null = combined_df[col].notna().sum()
                null_pct = 100 * (1 - non_null / len(combined_df))
                f.write(f"{col:30s} {str(dtype):15s} {non_null:>10,} non-null ({null_pct:5.1f}% null)\n")
        logger.info(f"✓ Saved column info: {column_info_path}")
        
        # Save completion marker
        logger.info("\n" + "-" * 70)
        logger.info("Saving completion marker...")
        logger.info("-" * 70)
        
        completion_file = step_results_dir / "step_03c_completed.txt"
        
        completion_message = f"""Step 03c - Load Comprehensive Data from All CSV Files
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Files:
- Quarter_1.csv, Quarter_2.csv, Quarter_3.csv, Quarter_4.csv
- RAW_DATA.csv

Processing Results:
- Total rows: {len(combined_df):,}
- Total columns: {len(combined_df.columns)}
- Unique companies: {combined_df['company_name'].nunique()}
- Unique features: {combined_df['feature_name'].nunique() if 'feature_name' in combined_df.columns else 0}

Data Coverage:
- Rows with earnings dates: {combined_df['earnings_date'].notna().sum():,} ({100*combined_df['earnings_date'].notna().sum()/len(combined_df):.1f}%)
- Non-null feature values: {combined_df['value'].notna().sum():,} ({100*combined_df['value'].notna().sum()/len(combined_df):.1f}%)

Output Files:
- {output_path.name} ({output_path.stat().st_size / 1024**2:.2f} MB)
- {sample_path.name}
- {column_info_path.name}

Status: SUCCESS
All data from CSV files successfully loaded and combined.
N/A values preserved (missing data is intentional).
Ready for further processing and analysis.
"""
        
        with open(completion_file, 'w') as f:
            f.write(completion_message)
        
        logger.info(f"✓ Completion marker saved: {completion_file}")
        
        # Final success message
        logger.info("\n" + "=" * 70)
        logger.info("Step 03c completed successfully: comprehensive data loaded")
        logger.info("=" * 70)
        logger.info(f"\nKey outputs:")
        logger.info(f"  - {output_path}")
        logger.info(f"  - {sample_path}")
        logger.info(f"  - {column_info_path}")
        logger.info(f"  - {completion_file}")
        logger.info(f"\nTotal rows: {len(combined_df):,}")
        logger.info(f"Companies: {combined_df['company_name'].nunique()}")
        logger.info(f"Features: {combined_df['feature_name'].nunique() if 'feature_name' in combined_df.columns else 0}")
        logger.info("\nData is ready for cleaning and transformation.")
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"\n✗ Step 03c failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    run_step_03c()
