"""
Step 05 — Fetch daily price/volume data via yfinance with local caching.

This step:
- Reads the earnings dataset from results/step_03/earnings_raw.parquet
- Determines which tickers and date ranges are needed
- Downloads daily OHLCV data for a limited set of tickers and for SPY
- Relies on a local cache under data/cache/yfinance to avoid re-downloading
- Writes a manifest summarizing what was downloaded

The downloaded data will be used in subsequent steps to compute pre- and
post-earnings returns and excess returns vs. SPY.

Usage
-----
    python -m src.step_05_fetch_daily_data
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, NoReturn

import pandas as pd

from src.config import settings
from src.live_data.get_pre_earnings import fetch_daily_data_for_tickers


def run_step_05() -> None:
    """
    Execute Step 05: Fetch daily price/volume data.
    
    This function orchestrates the download of daily OHLCV data for tickers
    appearing in the earnings dataset, using yfinance with local caching.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If earnings_raw.parquet from Step 03 is missing.
    Exception
        If any critical step fails.
    """
    # Initialize logging
    logger = settings.setup_logging("step_05_fetch_daily_data")
    logger.info("=" * 70)
    logger.info("STEP 05: FETCH DAILY PRICE/VOLUME DATA")
    logger.info("=" * 70)
    
    try:
        # Ensure directories exist
        settings.ensure_directories()
        
        # Create step-specific results directory
        step_results_dir = settings.get_step_results_dir(5)
        logger.info(f"\nStep 05 results directory: {step_results_dir}")
        logger.info(f"yfinance cache directory: {settings.YFINANCE_CACHE_DIR}")
        
        # Step 1: Load earnings dataset from Step 03 (clean data with tickers)
        logger.info("\n" + "-" * 70)
        logger.info("Loading earnings dataset from Step 03...")
        logger.info("-" * 70)
        
        # Try to load clean_data.parquet first (has ticker mappings)
        earnings_file = settings.get_step_results_dir(3) / "clean_data.parquet"
        
        if not earnings_file.exists():
            # Fall back to earnings_dates.parquet
            logger.warning("clean_data.parquet not found, trying earnings_dates.parquet")
            earnings_file = settings.get_step_results_dir(3) / "earnings_dates.parquet"
            
            if not earnings_file.exists():
                # Fall back to earnings_raw.parquet
                logger.warning("earnings_dates.parquet not found, trying earnings_raw.parquet")
                earnings_file = settings.get_step_results_dir(3) / "earnings_raw.parquet"
                
                if not earnings_file.exists():
                    raise FileNotFoundError(
                        f"Earnings data not found. "
                        f"Please run Step 03 first."
                    )
        
        earnings_df = pd.read_parquet(earnings_file)
        logger.info(f"✓ Loaded earnings data: {earnings_df.shape[0]:,} rows × {earnings_df.shape[1]} columns")
        
        # Filter to rows with tickers if available
        if 'ticker' in earnings_df.columns:
            initial_rows = len(earnings_df)
            earnings_df = earnings_df[earnings_df['ticker'].notna()].copy()
            logger.info(f"  Filtered to rows with tickers: {len(earnings_df):,} / {initial_rows:,} ({100*len(earnings_df)/initial_rows:.1f}%)")
        
        # Step 2: Fetch daily data for tickers
        logger.info("\n" + "-" * 70)
        logger.info("Fetching daily price/volume data...")
        logger.info("-" * 70)
        logger.info(f"Max tickers to download: {settings.MAX_TICKERS_DAILY_DOWNLOAD}")
        logger.info(f"Ticker column: {settings.EARNINGS_TICKER_COLUMN}")
        logger.info(f"Date column: {settings.EARNINGS_DATE_COLUMN}")
        
        daily_data: Dict[str, pd.DataFrame] = fetch_daily_data_for_tickers(
            earnings_df, settings
        )
        
        logger.info(f"\n✓ Downloaded data for {len(daily_data)} tickers")
        
        # Step 3: Save per-ticker Parquet files
        logger.info("\n" + "-" * 70)
        logger.info("Saving per-ticker Parquet files...")
        logger.info("-" * 70)
        
        manifest = {}
        
        for ticker, df in daily_data.items():
            if df.empty:
                logger.warning(f"  ⚠ Skipping {ticker} (empty DataFrame)")
                continue
            
            # Save to parquet
            output_file = step_results_dir / f"{ticker}_daily.parquet"
            df.to_parquet(output_file, engine='pyarrow')
            
            # Add to manifest
            manifest[ticker] = {
                "start": df.index.min().strftime('%Y-%m-%d'),
                "end": df.index.max().strftime('%Y-%m-%d'),
                "rows": len(df),
                "file": output_file.name
            }
            
            logger.info(f"  ✓ Saved {ticker}: {len(df):,} rows ({df.index.min().date()} to {df.index.max().date()})")
        
        # Step 4: Save manifest
        logger.info("\n" + "-" * 70)
        logger.info("Saving manifest...")
        logger.info("-" * 70)
        
        manifest_file = step_results_dir / "daily_data_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✓ Manifest saved: {manifest_file}")
        logger.info(f"  Total tickers in manifest: {len(manifest)}")
        
        # Step 5: Data quality summary
        logger.info("\n" + "-" * 70)
        logger.info("Data quality summary...")
        logger.info("-" * 70)
        
        total_rows = sum(info['rows'] for info in manifest.values())
        logger.info(f"  Total data points downloaded: {total_rows:,}")
        
        if settings.SPY_TICKER in manifest:
            spy_info = manifest[settings.SPY_TICKER]
            logger.info(f"  SPY benchmark: {spy_info['rows']:,} days ({spy_info['start']} to {spy_info['end']})")
        
        # Check cache usage
        cache_files = list(settings.YFINANCE_CACHE_DIR.glob("*.parquet"))
        logger.info(f"  Cache files: {len(cache_files)}")
        
        # Step 6: Save completion marker
        logger.info("\n" + "-" * 70)
        logger.info("Saving completion marker...")
        logger.info("-" * 70)
        
        completion_file = step_results_dir / "step_05_completed.txt"
        
        completion_message = f"""Step 05 - Fetch Daily Price/Volume Data
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Download Summary:
- Tickers processed: {len(manifest)}
- Total data points: {total_rows:,}
- Cache directory: {settings.YFINANCE_CACHE_DIR}
- Cache files: {len(cache_files)}

Configuration:
- Max tickers: {settings.MAX_TICKERS_DAILY_DOWNLOAD}
- Ticker column: {settings.EARNINGS_TICKER_COLUMN}
- Date column: {settings.EARNINGS_DATE_COLUMN}

Output Files:
- Per-ticker Parquet files: {len(manifest)} files
- Manifest: {manifest_file.name}

Status: SUCCESS
Daily price/volume data successfully downloaded and cached.
"""
        
        with open(completion_file, 'w') as f:
            f.write(completion_message)
        
        logger.info(f"✓ Completion marker saved: {completion_file}")
        
        # Final success message
        logger.info("\n" + "=" * 70)
        logger.info("Step 05 completed successfully: daily data downloaded and cached")
        logger.info("=" * 70)
        logger.info(f"\nKey outputs:")
        logger.info(f"  - {len(manifest)} ticker Parquet files in {step_results_dir}")
        logger.info(f"  - {manifest_file}")
        logger.info(f"  - {completion_file}")
        logger.info(f"\nCache location: {settings.YFINANCE_CACHE_DIR}")
        logger.info(f"  ({len(cache_files)} cached files)")
        logger.info("\nYou may proceed to Step 06 when ready.")
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"\n✗ Step 05 failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    run_step_05()
