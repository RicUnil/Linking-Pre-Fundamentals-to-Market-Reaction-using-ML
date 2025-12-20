"""
Step 09 — Build market-based features.

This step:
- Loads the earnings dataset with targets and fundamental features from Step 08.
- Uses daily price/volume data from Step 05 (stock + SPY).
- Computes market features such as momentum, volatility, pre-earnings volume,
  and SPY pre-earnings return.
- Saves an enriched dataset ready for cleaning and modelling.
"""

from typing import NoReturn
import logging

import pandas as pd

from src.config import Settings
from src.preprocessing.features import add_market_features


def run_step_09() -> NoReturn:
    """
    Execute Step 09: Build market-based features.
    
    This step loads the earnings dataset with targets and fundamental features
    from Step 08, computes market-based features using daily price/volume data,
    and saves the enriched dataset for subsequent steps.
    """
    # Initialize settings
    settings = Settings()
    logger = settings.setup_logging("step_09_build_market_features")
    
    logger.info("=" * 70)
    logger.info("STEP 09: BUILD MARKET-BASED FEATURES")
    logger.info("=" * 70)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Create step 09 results directory
    output_dir = settings.RESULTS_DIR / "step_09"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nStep 09 results directory: {output_dir}")
    
    # ========================================================================
    # Load Step 08 output
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Loading earnings dataset from Step 08...")
    logger.info("-" * 70)
    
    input_path = settings.RESULTS_DIR / "step_08" / "earnings_with_targets_and_fundamentals.parquet"
    
    if not input_path.exists():
        error_msg = (
            f"Step 08 output not found at {input_path}. "
            f"Please run Step 08 first:\n"
            f"  python -m src.step_08_build_fundamental_features"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"  Loading from: {input_path}")
    df = pd.read_parquet(input_path)
    
    logger.info(f"✓ Loaded earnings data: {len(df):,} rows × {len(df.columns)} columns")
    
    # ========================================================================
    # Add market features
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Computing market-based features...")
    logger.info("-" * 70)
    
    enriched = add_market_features(df, settings)
    
    # Identify new market feature columns
    market_feature_cols = [
        "stock_momentum_1m",
        "stock_momentum_3m",
        "stock_momentum_6m",
        "pre_volatility_30d",
        "pre_avg_volume_30d",
        "spy_pre_return_30d",
    ]
    
    logger.info(f"\n✓ Market features added: {len(market_feature_cols)} new columns")
    logger.info(f"  Feature columns: {market_feature_cols}")
    
    # ========================================================================
    # Data quality summary
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Data quality summary...")
    logger.info("-" * 70)
    
    for col in market_feature_cols:
        non_null = enriched[col].notna().sum()
        pct = 100 * non_null / len(enriched)
        logger.info(f"  {col}: {non_null:,} non-null ({pct:.1f}%)")
    
    # ========================================================================
    # Save enriched dataset
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving enriched dataset...")
    logger.info("-" * 70)
    
    full_path = output_dir / "earnings_with_all_features.parquet"
    head_path = output_dir / "earnings_with_all_features_head.csv"
    marker_path = output_dir / "step_09_completed.txt"
    
    # Save full dataset
    enriched.to_parquet(full_path, index=False)
    logger.info(f"✓ Saved enriched dataset: {full_path}")
    logger.info(f"  Rows: {len(enriched):,}")
    logger.info(f"  Columns: {len(enriched.columns)}")
    logger.info(f"  File size: {full_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save sample
    enriched.head(50).to_csv(head_path, index=False)
    logger.info(f"✓ Saved sample (first 50 rows): {head_path}")
    
    # ========================================================================
    # Save completion marker
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving completion marker...")
    logger.info("-" * 70)
    
    with marker_path.open("w", encoding="utf-8") as f:
        f.write(f"Step 09 completed.\n")
        f.write(f"Rows: {enriched.shape[0]}\n")
        f.write(f"Columns: {enriched.shape[1]}\n")
        f.write(f"Market features: {len(market_feature_cols)}\n")
    
    logger.info(f"✓ Completion marker saved: {marker_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 09 completed successfully: market-based features computed")
    logger.info("=" * 70)
    logger.info("\nKey outputs:")
    logger.info(f"  - {full_path}")
    logger.info(f"  - {head_path}")
    logger.info(f"  - {marker_path}")
    logger.info(f"\nEvents processed: {len(enriched):,}")
    logger.info(f"Market features added: {len(market_feature_cols)}")
    
    # Count features with data
    features_with_data = sum(1 for col in market_feature_cols if enriched[col].notna().sum() > 0)
    logger.info(f"Features with data: {features_with_data}/{len(market_feature_cols)}")
    
    logger.info("\nYou may proceed to Step 10 when ready.")


if __name__ == "__main__":
    run_step_09()
