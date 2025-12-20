"""
Step 10 — Data cleaning, preprocessing and train/val/test split.

This step:
- Loads the enriched dataset with targets, fundamentals and market features
  from Step 09.
- Defines a clear feature set and dataset specification.
- Cleans and filters rows (valid targets, post-window coverage).
- Performs time-based train/val/test split:
    * Train & val: earnings_date < 2020-01-01
    * Test:        earnings_date >= 2020-01-01
- Applies scaling (StandardScaler) fitted on the training set.
- Saves cleaned dataframes, scaled matrices and metadata for modelling steps.
"""

from typing import NoReturn
import logging

import pandas as pd

from src.config import Settings
from src.preprocessing.dataset import (
    build_dataset_spec,
    clean_and_filter_rows,
    train_val_test_split_time_based,
    scale_and_package_matrices,
    save_preprocessed_data,
)


def run_step_10() -> NoReturn:
    """
    Execute Step 10: Prepare modelling data with cleaning, splitting, and scaling.
    
    This step loads the enriched dataset from Step 09, defines the feature set,
    cleans rows, performs time-based train/val/test split, applies scaling,
    and saves all preprocessed artifacts for subsequent modelling steps.
    """
    # Initialize settings
    settings = Settings()
    logger = settings.setup_logging("step_10_prepare_modelling_data")
    
    logger.info("=" * 70)
    logger.info("STEP 10: PREPARE MODELLING DATA")
    logger.info("=" * 70)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Define paths
    input_path = settings.RESULTS_DIR / "step_09" / "earnings_with_all_features.parquet"
    output_dir = settings.RESULTS_DIR / "step_10"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nInput: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # ========================================================================
    # Load Step 09 output
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Loading enriched dataset from Step 09...")
    logger.info("-" * 70)
    
    if not input_path.exists():
        error_msg = (
            f"Input file not found at {input_path}. "
            f"Please run Step 09 first:\n"
            f"  python -m src.step_09_build_market_features"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    df = pd.read_parquet(input_path)
    logger.info(f"✓ Loaded dataset: {len(df):,} rows × {len(df.columns)} columns")
    
    # ========================================================================
    # Build dataset specification
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Building dataset specification...")
    logger.info("-" * 70)
    
    spec = build_dataset_spec(df, settings)
    
    # ========================================================================
    # Clean and filter rows
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Cleaning and filtering rows...")
    logger.info("-" * 70)
    
    df_clean = clean_and_filter_rows(df, spec)
    
    # ========================================================================
    # Time-based train/val/test split
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Creating time-based train/val/test split...")
    logger.info("-" * 70)
    
    splits = train_val_test_split_time_based(
        df_clean,
        spec,
        settings,
        test_split_date="2020-01-01",
        val_fraction_within_train=0.2,
    )
    
    # ========================================================================
    # Scale and package into matrices
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Scaling features and packaging matrices...")
    logger.info("-" * 70)
    
    X_splits, y_splits, scaler = scale_and_package_matrices(splits, spec)
    
    # ========================================================================
    # Save all preprocessed data
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving preprocessed data...")
    logger.info("-" * 70)
    
    save_preprocessed_data(splits, X_splits, y_splits, scaler, spec, output_dir)
    
    # ========================================================================
    # Save completion marker
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving completion marker...")
    logger.info("-" * 70)
    
    marker_path = output_dir / "step_10_completed.txt"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write(f"Step 10 completed.\n")
        f.write(f"Train rows: {splits['train'].shape[0]}\n")
        f.write(f"Val rows: {splits['val'].shape[0]}\n")
        f.write(f"Test rows: {splits['test'].shape[0]}\n")
        f.write(f"Features: {len(spec.feature_columns)}\n")
        f.write(f"Test split date: 2020-01-01\n")
    
    logger.info(f"✓ Completion marker saved: {marker_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 10 completed successfully: data cleaned, split, and scaled")
    logger.info("=" * 70)
    
    logger.info("\nKey outputs:")
    logger.info(f"  - Cleaned dataframes: cleaned_{{train,val,test}}.parquet")
    logger.info(f"  - Feature matrices: X_{{train,val,test}}.npy")
    logger.info(f"  - Target arrays: y_{{train,val,test}}.npy")
    logger.info(f"  - Scaler: scaler.joblib")
    logger.info(f"  - Metadata: dataset_spec.json, split_summary.json")
    
    logger.info(f"\nDataset summary:")
    logger.info(f"  Features: {len(spec.feature_columns)}")
    logger.info(f"  Train: {len(splits['train']):,} events")
    logger.info(f"  Val:   {len(splits['val']):,} events")
    logger.info(f"  Test:  {len(splits['test']):,} events")
    logger.info(f"  Total: {len(df_clean):,} events")
    
    logger.info("\nData is ready for modelling (Steps 11+).")
    logger.info("You may proceed to Step 11 when ready.")


if __name__ == "__main__":
    run_step_10()
