"""
Step 07 — Compute 30-day post-earnings returns and excess returns (targets).

This step:
- Reads the enriched earnings dataset with windows from Step 06
- Uses daily price data from Step 05 (stock + SPY)
- Computes:
    * stock_return_30d: Stock's 30-day return after earnings
    * spy_return_30d: SPY's 30-day return over the same window
    * excess_return_30d: Stock return minus SPY return (alpha)
    * label_outperform_30d: Binary label (1 if excess > 0, else 0)
- Saves the resulting dataset for later feature engineering and modeling

This is the target construction step. No features or models are built yet.

Usage
-----
    python -m src.step_07_compute_targets
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import pandas as pd

from src.config import settings
from src.preprocessing.targets import add_post_earnings_targets


def run_step_07() -> None:
    """
    Execute Step 07: Compute 30-day post-earnings targets.
    
    This function orchestrates the computation of post-earnings returns,
    excess returns vs SPY, and binary outperformance labels.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If required input files from Steps 05 or 06 are missing.
    Exception
        If any critical step fails.
    """
    # Initialize logging
    logger = settings.setup_logging("step_07_compute_targets")
    logger.info("=" * 70)
    logger.info("STEP 07: COMPUTE 30-DAY POST-EARNINGS TARGETS")
    logger.info("=" * 70)
    
    try:
        # Ensure directories exist
        settings.ensure_directories()
        
        # Create step-specific results directory
        step_results_dir = settings.get_step_results_dir(7)
        logger.info(f"\nStep 07 results directory: {step_results_dir}")
        
        # Step 1: Load earnings dataset with windows from Step 06
        logger.info("\n" + "-" * 70)
        logger.info("Loading earnings dataset from Step 06...")
        logger.info("-" * 70)
        
        input_path = settings.get_step_results_dir(6) / "earnings_with_windows.parquet"
        
        if not input_path.exists():
            raise FileNotFoundError(
                f"Earnings with windows not found at {input_path}. "
                f"Please run Step 06 first."
            )
        
        earnings_df = pd.read_parquet(input_path)
        logger.info(f"✓ Loaded earnings data: {earnings_df.shape[0]:,} rows × {earnings_df.shape[1]} columns")
        
        # Check for required columns
        required_cols = ['post_window_start', 'post_window_end', 'has_full_post_window']
        missing_cols = [col for col in required_cols if col not in earnings_df.columns]
        
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")
            logger.warning("This may affect target computation")
        
        # Step 2: Verify daily data availability
        logger.info("\n" + "-" * 70)
        logger.info("Verifying daily data availability...")
        logger.info("-" * 70)
        
        daily_data_dir = settings.get_step_results_dir(5)
        
        # Check for consolidated data first
        consolidated_path = daily_data_dir / "cache" / "all_daily_data.parquet"
        
        if consolidated_path.exists():
            logger.info(f"✓ Found consolidated daily data: {consolidated_path}")
            # Verify it has SPY
            df_check = pd.read_parquet(consolidated_path)
            if settings.SPY_TICKER in df_check['ticker'].values:
                logger.info(f"  ✓ SPY data available in consolidated file")
                logger.info(f"  Available tickers: {df_check['ticker'].nunique()}")
            else:
                raise FileNotFoundError(
                    f"SPY data not found in consolidated file. "
                    f"Please run Step 05 first."
                )
        else:
            # Check for individual files
            spy_path = daily_data_dir / f"{settings.SPY_TICKER}_daily.parquet"
            
            if not spy_path.exists():
                raise FileNotFoundError(
                    f"SPY daily data not found at {spy_path}. "
                    f"Please run Step 05 first."
                )
            
            # Count available ticker files
            ticker_files = list(daily_data_dir.glob("*_daily.parquet"))
            logger.info(f"✓ Daily data directory: {daily_data_dir}")
            logger.info(f"  Available ticker files: {len(ticker_files)}")
        
        # Step 3: Compute targets
        logger.info("\n" + "-" * 70)
        logger.info("Computing post-earnings targets...")
        logger.info("-" * 70)
        logger.info(f"Configuration:")
        logger.info(f"  Ticker column: {settings.EARNINGS_TICKER_COLUMN}")
        logger.info(f"  SPY ticker: {settings.SPY_TICKER}")
        logger.info(f"  Window: post_window_start to post_window_end (J+1 to J+30)")
        
        enriched_df = add_post_earnings_targets(earnings_df, settings)
        
        logger.info(f"\n✓ Targets computed for {len(enriched_df):,} events")
        
        # Step 4: Data quality summary
        logger.info("\n" + "-" * 70)
        logger.info("Data quality summary...")
        logger.info("-" * 70)
        
        # Check for new columns
        new_columns = [
            'stock_return_30d',
            'spy_return_30d',
            'excess_return_30d',
            'label_outperform_30d'
        ]
        
        for col in new_columns:
            if col in enriched_df.columns:
                non_null = enriched_df[col].notna().sum()
                logger.info(f"  ✓ Column added: {col} ({non_null:,} non-null values)")
        
        # Compute statistics on valid targets
        valid_excess = enriched_df['excess_return_30d'].notna().sum()
        total_events = len(enriched_df)
        
        logger.info(f"\n  Target Availability:")
        logger.info(f"  Total events: {total_events:,}")
        logger.info(f"  Events with valid excess return: {valid_excess:,} ({100*valid_excess/total_events:.1f}%)")
        
        # Step 5: Save enriched dataset
        logger.info("\n" + "-" * 70)
        logger.info("Saving enriched dataset with targets...")
        logger.info("-" * 70)
        
        # Save full dataset as parquet
        output_path = step_results_dir / "earnings_with_targets.parquet"
        enriched_df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"✓ Saved enriched dataset: {output_path}")
        logger.info(f"  Rows: {len(enriched_df):,}")
        logger.info(f"  Columns: {len(enriched_df.columns)}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")
        
        # Save sample as CSV
        sample_path = step_results_dir / "earnings_with_targets_head.csv"
        
        # Select relevant columns for the sample
        sample_cols = [settings.EARNINGS_TICKER_COLUMN, settings.EARNINGS_DATE_COLUMN]
        sample_cols.extend([
            'post_window_start', 'post_window_end',
            'stock_return_30d', 'spy_return_30d',
            'excess_return_30d', 'label_outperform_30d'
        ])
        
        # Only include columns that exist
        sample_cols = [col for col in sample_cols if col in enriched_df.columns]
        
        if sample_cols:
            enriched_df.head(30)[sample_cols].to_csv(sample_path, index=False)
            logger.info(f"✓ Saved sample (first 30 rows): {sample_path}")
        
        # Step 6: Save completion marker
        logger.info("\n" + "-" * 70)
        logger.info("Saving completion marker...")
        logger.info("-" * 70)
        
        completion_file = step_results_dir / "step_07_completed.txt"
        
        # Compute label distribution for summary
        valid_labels = enriched_df['label_outperform_30d'].notna()
        if valid_labels.sum() > 0:
            outperform_count = (enriched_df.loc[valid_labels, 'label_outperform_30d'] == 1).sum()
            underperform_count = (enriched_df.loc[valid_labels, 'label_outperform_30d'] == 0).sum()
        else:
            outperform_count = 0
            underperform_count = 0
        
        completion_message = f"""Step 07 - Compute 30-Day Post-Earnings Targets
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Data:
- Earnings events: {len(earnings_df):,} rows
- Daily data files: {len(ticker_files)}

Target Configuration:
- Window: post_window_start to post_window_end (J+1 to J+30)
- Return metric: Adjusted close (or close if unavailable)
- Excess return: stock_return_30d - spy_return_30d

Processing Results:
- Events processed: {len(enriched_df):,}
- Events with valid targets: {valid_excess:,} ({100*valid_excess/total_events:.1f}%)
- New columns added: {len(new_columns)}

Label Distribution:
- Outperform (label=1): {outperform_count:,} ({100*outperform_count/valid_labels.sum():.1f}% of valid)
- Underperform (label=0): {underperform_count:,} ({100*underperform_count/valid_labels.sum():.1f}% of valid)

Output Files:
- {output_path.name} ({output_path.stat().st_size / 1024**2:.2f} MB)
- {sample_path.name}

Status: SUCCESS
30-day post-earnings targets successfully computed.
Next step: Feature engineering (Step 08).
"""
        
        with open(completion_file, 'w') as f:
            f.write(completion_message)
        
        logger.info(f"✓ Completion marker saved: {completion_file}")
        
        # Final success message
        logger.info("\n" + "=" * 70)
        logger.info("Step 07 completed successfully: 30-day post-earnings targets computed")
        logger.info("=" * 70)
        logger.info(f"\nKey outputs:")
        logger.info(f"  - {output_path}")
        logger.info(f"  - {sample_path}")
        logger.info(f"  - {completion_file}")
        logger.info(f"\nValid targets: {valid_excess:,} / {total_events:,} events")
        logger.info("\nYou may proceed to Step 08 when ready.")
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"\n✗ Step 07 failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    run_step_07()
