"""
Step 06 — Map earnings events to daily data windows.

This step:
- Reads the raw earnings dataset from Step 03
- Computes pre- and post-earnings date windows (J-30..J-1, J+1..J+30)
- Uses the daily data manifest from Step 05 to flag whether each event
  has full pre/post coverage in the downloaded daily series
- Saves an enriched earnings dataset with window columns and coverage flags

This is a purely temporal mapping step. No returns, features, or targets
are computed yet.

Usage
-----
    python -m src.step_06_map_earnings_to_daily
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, NoReturn

import pandas as pd

from src.config import settings
from src.data.mapping import compute_event_windows, flag_window_coverage


def run_step_06() -> None:
    """
    Execute Step 06: Map earnings events to daily data windows.
    
    This function orchestrates the temporal mapping between earnings events
    and the daily price data downloaded in Step 05.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If required input files from Steps 03 or 05 are missing.
    Exception
        If any critical step fails.
    """
    # Initialize logging
    logger = settings.setup_logging("step_06_map_earnings_to_daily")
    logger.info("=" * 70)
    logger.info("STEP 06: MAP EARNINGS EVENTS TO DAILY DATA WINDOWS")
    logger.info("=" * 70)
    
    try:
        # Ensure directories exist
        settings.ensure_directories()
        
        # Create step-specific results directory
        step_results_dir = settings.get_step_results_dir(6)
        logger.info(f"\nStep 06 results directory: {step_results_dir}")
        
        # Step 1: Load earnings dataset from Step 03 (clean data with tickers)
        logger.info("\n" + "-" * 70)
        logger.info("Loading earnings dataset from Step 03...")
        logger.info("-" * 70)
        
        # Try to load clean_data_fixed.parquet first (Step 03e - properly aligned features)
        earnings_path = settings.get_step_results_dir(3) / "clean_data_fixed.parquet"
        
        if not earnings_path.exists():
            # Fall back to clean_data.parquet
            logger.warning("clean_data_fixed.parquet not found, trying clean_data.parquet")
            earnings_path = settings.get_step_results_dir(3) / "clean_data.parquet"
            
            if not earnings_path.exists():
                # Fall back to earnings_dates.parquet
                logger.warning("clean_data.parquet not found, trying earnings_dates.parquet")
                earnings_path = settings.get_step_results_dir(3) / "earnings_dates.parquet"
                
                if not earnings_path.exists():
                    # Fall back to earnings_raw.parquet
                    logger.warning("earnings_dates.parquet not found, trying earnings_raw.parquet")
                    earnings_path = settings.get_step_results_dir(3) / "earnings_raw.parquet"
                    
                    if not earnings_path.exists():
                        raise FileNotFoundError(
                            f"Earnings data not found. "
                            f"Please run Step 03 first."
                        )
        
        earnings_df = pd.read_parquet(earnings_path)
        logger.info(f"✓ Loaded earnings data: {earnings_df.shape[0]:,} rows × {earnings_df.shape[1]} columns")
        
        # Filter to rows with tickers and earnings dates if available
        if 'ticker' in earnings_df.columns and 'earnings_date' in earnings_df.columns:
            initial_rows = len(earnings_df)
            earnings_df = earnings_df[
                (earnings_df['ticker'].notna()) & 
                (earnings_df['earnings_date'].notna())
            ].copy()
            logger.info(f"  Filtered to rows with tickers and dates: {len(earnings_df):,} / {initial_rows:,} ({100*len(earnings_df)/initial_rows:.1f}%)")
        
        # Step 2: Load daily data manifest from Step 05
        logger.info("\n" + "-" * 70)
        logger.info("Loading daily data manifest from Step 05...")
        logger.info("-" * 70)
        
        manifest_path = settings.get_step_results_dir(5) / "daily_data_manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Daily data manifest not found at {manifest_path}. "
                f"Please run Step 05 first."
            )
        
        with open(manifest_path, 'r') as f:
            manifest: Dict[str, Dict] = json.load(f)
        
        logger.info(f"✓ Loaded manifest with {len(manifest)} tickers")
        logger.info(f"  Tickers: {list(manifest.keys())[:10]}{'...' if len(manifest) > 10 else ''}")
        
        # Step 3: Compute event windows
        logger.info("\n" + "-" * 70)
        logger.info("Computing pre- and post-earnings windows...")
        logger.info("-" * 70)
        logger.info(f"Configuration:")
        logger.info(f"  Ticker column: {settings.EARNINGS_TICKER_COLUMN}")
        logger.info(f"  Date column: {settings.EARNINGS_DATE_COLUMN}")
        logger.info(f"  Pre-window: J-30 to J-1")
        logger.info(f"  Post-window: J+1 to J+30")
        
        earnings_with_windows = compute_event_windows(earnings_df, settings)
        
        logger.info(f"\n✓ Windows computed for {len(earnings_with_windows):,} events")
        
        # Step 4: Flag window coverage
        logger.info("\n" + "-" * 70)
        logger.info("Flagging window coverage...")
        logger.info("-" * 70)
        
        earnings_enriched = flag_window_coverage(
            earnings_with_windows, manifest, settings
        )
        
        logger.info(f"\n✓ Coverage flags added to {len(earnings_enriched):,} events")
        
        # Step 5: Data quality summary
        logger.info("\n" + "-" * 70)
        logger.info("Data quality summary...")
        logger.info("-" * 70)
        
        # Check for new columns
        new_columns = [
            'pre_window_start', 'pre_window_end',
            'post_window_start', 'post_window_end',
            'has_full_pre_window', 'has_full_post_window'
        ]
        
        for col in new_columns:
            if col in earnings_enriched.columns:
                logger.info(f"  ✓ Column added: {col}")
        
        # Coverage statistics
        if 'has_full_pre_window' in earnings_enriched.columns:
            pre_count = earnings_enriched['has_full_pre_window'].sum()
            post_count = earnings_enriched['has_full_post_window'].sum()
            both_count = (
                earnings_enriched['has_full_pre_window'] & 
                earnings_enriched['has_full_post_window']
            ).sum()
            
            logger.info(f"\n  Final Coverage Statistics:")
            logger.info(f"  Events with full pre-window: {pre_count:,}")
            logger.info(f"  Events with full post-window: {post_count:,}")
            logger.info(f"  Events with both windows: {both_count:,}")
        
        # Step 6: Save enriched dataset
        logger.info("\n" + "-" * 70)
        logger.info("Saving enriched earnings dataset...")
        logger.info("-" * 70)
        
        # Save full dataset as parquet
        output_path = step_results_dir / "earnings_with_windows.parquet"
        earnings_enriched.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"✓ Saved enriched dataset: {output_path}")
        logger.info(f"  Rows: {len(earnings_enriched):,}")
        logger.info(f"  Columns: {len(earnings_enriched.columns)}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")
        
        # Save sample as CSV
        sample_path = step_results_dir / "earnings_with_windows_head.csv"
        sample_df = earnings_enriched.head(20)
        
        # Select relevant columns for the sample
        sample_cols = [settings.EARNINGS_TICKER_COLUMN, settings.EARNINGS_DATE_COLUMN]
        sample_cols.extend(new_columns)
        
        # Only include columns that exist
        sample_cols = [col for col in sample_cols if col in earnings_enriched.columns]
        
        if sample_cols:
            sample_df[sample_cols].to_csv(sample_path, index=False)
            logger.info(f"✓ Saved sample (first 20 rows): {sample_path}")
        
        # Step 7: Save completion marker
        logger.info("\n" + "-" * 70)
        logger.info("Saving completion marker...")
        logger.info("-" * 70)
        
        completion_file = step_results_dir / "step_06_completed.txt"
        
        completion_message = f"""Step 06 - Map Earnings Events to Daily Data Windows
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Data:
- Earnings events: {len(earnings_df):,} rows
- Daily data tickers: {len(manifest)}

Window Configuration:
- Pre-window: J-30 to J-1 (30 trading days before earnings)
- Post-window: J+1 to J+30 (30 trading days after earnings)

Processing Results:
- Events processed: {len(earnings_enriched):,}
- New columns added: {len(new_columns)}

Coverage Statistics:
- Events with full pre-window: {pre_count:,} ({100*pre_count/len(earnings_enriched):.1f}%)
- Events with full post-window: {post_count:,} ({100*post_count/len(earnings_enriched):.1f}%)
- Events with both windows: {both_count:,} ({100*both_count/len(earnings_enriched):.1f}%)

Output Files:
- {output_path.name} ({output_path.stat().st_size / 1024**2:.2f} MB)
- {sample_path.name}

Status: SUCCESS
Earnings events successfully mapped to daily data windows.
Next step: Compute returns and targets (Step 07).
"""
        
        with open(completion_file, 'w') as f:
            f.write(completion_message)
        
        logger.info(f"✓ Completion marker saved: {completion_file}")
        
        # Final success message
        logger.info("\n" + "=" * 70)
        logger.info("Step 06 completed successfully: windows computed and coverage flagged")
        logger.info("=" * 70)
        logger.info(f"\nKey outputs:")
        logger.info(f"  - {output_path}")
        logger.info(f"  - {sample_path}")
        logger.info(f"  - {completion_file}")
        logger.info(f"\nEvents with full coverage: {both_count:,} / {len(earnings_enriched):,}")
        logger.info("\nYou may proceed to Step 07 when ready.")
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"\n✗ Step 06 failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    run_step_06()
