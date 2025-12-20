"""
Step 08 — Build fundamental features.

This step:
- Loads the earnings dataset with targets from Step 07.
- Computes fundamental accounting-based features at the earnings-event level.
- Saves an enriched dataset for subsequent market feature engineering and modelling.
"""

from typing import NoReturn
import logging
from pathlib import Path

import pandas as pd

from src.config import Settings
from src.preprocessing.features import add_fundamental_features


def run_step_08() -> NoReturn:
    """
    Execute Step 08: Build fundamental features.
    
    This step loads the earnings dataset with targets from Step 07,
    computes fundamental accounting-based features, and saves the
    enriched dataset for subsequent steps.
    """
    # Initialize settings
    settings = Settings()
    logger = settings.setup_logging("step_08_build_fundamental_features")
    
    logger.info("=" * 70)
    logger.info("STEP 08: BUILD FUNDAMENTAL FEATURES")
    logger.info("=" * 70)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Create step 08 results directory
    step_08_dir = settings.get_step_results_dir(8)
    step_08_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nStep 08 results directory: {step_08_dir}")
    
    # ========================================================================
    # Load Step 07 output
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Loading earnings dataset...")
    logger.info("-" * 70)
    
    # Try Step 07 output first (has targets)
    input_path = settings.RESULTS_DIR / "step_07" / "earnings_with_targets.parquet"
    
    if input_path.exists():
        logger.info(f"  Loading from Step 07: {input_path}")
        df = pd.read_parquet(input_path)
        logger.info(f"✓ Loaded earnings data with targets: {len(df):,} rows × {len(df.columns)} columns")
    else:
        # Fall back to Step 06 output (has fundamental features but no targets)
        logger.warning("Step 07 output not found, loading from Step 06...")
        input_path = settings.RESULTS_DIR / "step_06" / "earnings_with_windows.parquet"
        
        if not input_path.exists():
            error_msg = (
                f"Neither Step 07 nor Step 06 output found. "
                f"Please run Step 06 or Step 07 first."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"  Loading from Step 06: {input_path}")
        df = pd.read_parquet(input_path)
        logger.info(f"✓ Loaded earnings data: {len(df):,} rows × {len(df.columns)} columns")
        logger.info("  Note: Targets not available (Step 07 not run yet)")
    
    logger.info(f"  Columns: {list(df.columns)}")
    
    # ========================================================================
    # Add fundamental features
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Computing fundamental features...")
    logger.info("-" * 70)
    
    enriched = add_fundamental_features(df, settings)
    
    # Identify new feature columns (those starting with 'f_')
    feature_cols = [col for col in enriched.columns if col.startswith('f_')]
    
    logger.info(f"\n✓ Features computed: {len(feature_cols)} new columns")
    logger.info(f"  Feature columns: {feature_cols}")
    
    # ========================================================================
    # Data quality summary
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Data quality summary...")
    logger.info("-" * 70)
    
    for col in feature_cols:
        non_null_count = enriched[col].notna().sum()
        non_null_pct = 100 * non_null_count / len(enriched)
        logger.info(f"  {col}: {non_null_count:,} non-null ({non_null_pct:.1f}%)")
    
    # ========================================================================
    # Save enriched dataset
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving enriched dataset...")
    logger.info("-" * 70)
    
    # Save full dataset
    output_path = step_08_dir / "earnings_with_targets_and_fundamentals.parquet"
    enriched.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    
    file_size_mb = output_path.stat().st_size / (1024**2)
    
    logger.info(f"✓ Saved enriched dataset: {output_path}")
    logger.info(f"  Rows: {len(enriched):,}")
    logger.info(f"  Columns: {len(enriched.columns)}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    
    # Save sample (head)
    sample_path = step_08_dir / "earnings_with_targets_and_fundamentals_head.csv"
    enriched.head(30).to_csv(sample_path, index=False)
    
    logger.info(f"✓ Saved sample (first 30 rows): {sample_path}")
    
    # ========================================================================
    # Save completion marker
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving completion marker...")
    logger.info("-" * 70)
    
    completion_path = step_08_dir / "step_08_completed.txt"
    
    # Count features with data
    features_with_data = sum(1 for col in feature_cols if enriched[col].notna().sum() > 0)
    
    # Get a key feature for summary
    key_feature = 'f_revenue_growth_qoq'
    key_feature_count = enriched[key_feature].notna().sum() if key_feature in enriched.columns else 0
    
    with open(completion_path, 'w') as f:
        f.write("Step 08: Build Fundamental Features\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Status: COMPLETED\n\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Output: {output_path}\n\n")
        f.write(f"Events processed: {len(enriched):,}\n")
        f.write(f"Features added: {len(feature_cols)}\n")
        f.write(f"Features with data: {features_with_data}/{len(feature_cols)}\n\n")
        f.write(f"Example feature ({key_feature}): {key_feature_count:,} non-null values\n\n")
        f.write("Feature columns:\n")
        for col in feature_cols:
            non_null = enriched[col].notna().sum()
            f.write(f"  - {col}: {non_null:,} non-null\n")
    
    logger.info(f"✓ Completion marker saved: {completion_path}")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 08 completed successfully: fundamental features computed")
    logger.info("=" * 70)
    logger.info("\nKey outputs:")
    logger.info(f"  - {output_path}")
    logger.info(f"  - {sample_path}")
    logger.info(f"  - {completion_path}")
    logger.info(f"\nEvents processed: {len(enriched):,}")
    logger.info(f"Features added: {len(feature_cols)}")
    logger.info(f"Features with data: {features_with_data}/{len(feature_cols)}")
    logger.info("\nYou may proceed to Step 09 when ready.")


if __name__ == "__main__":
    run_step_08()
