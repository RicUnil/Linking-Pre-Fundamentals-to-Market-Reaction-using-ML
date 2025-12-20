"""
Dataset preparation utilities for ML modelling.

This module provides functions to:
- Define dataset specifications (features, targets, identifiers)
- Clean and filter rows for modelling
- Perform time-based train/validation/test splits
- Scale features using StandardScaler
- Package and save preprocessed data
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import Settings


logger = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
    """
    Specification of the modelling dataset: which columns are used as features,
    which are targets, and which are identifiers.
    
    Attributes
    ----------
    id_columns : List[str]
        Columns used for identification (ticker, date, etc.).
    feature_columns : List[str]
        Columns used as predictive features.
    target_regression : str
        Column name for regression target (default: excess_return_30d).
    target_classification : str
        Column name for classification target (default: label_outperform_30d).
    """
    
    id_columns: List[str]
    feature_columns: List[str]
    target_regression: str = "excess_return_30d"
    target_classification: str = "label_outperform_30d"


def build_dataset_spec(df: pd.DataFrame, settings: Settings) -> DatasetSpec:
    """
    Build a DatasetSpec object listing identifier, feature, and target columns.

    This function:
    - Always treats the ticker and earnings date as identifiers.
    - Excludes obvious non-feature columns (windows, flags, raw returns).
    - Uses all f_* fundamental features and market features as predictors.
    
    Parameters
    ----------
    df : pd.DataFrame
        The full dataset with all columns.
    settings : Settings
        Project settings containing column name configurations.
    
    Returns
    -------
    DatasetSpec
        Dataset specification with id, feature, and target columns defined.
    """
    logger.info("Building dataset specification...")
    
    # Identifier columns
    ticker_col = settings.EARNINGS_TICKER_COLUMN
    date_col = settings.EARNINGS_DATE_COLUMN
    
    id_columns = [ticker_col, date_col]
    
    # Add other identifier columns if present
    for col in ["CUSIP", "GVKEY", "isin_code", "company_name"]:
        if col in df.columns and col not in id_columns:
            id_columns.append(col)
    
    # Columns to exclude from features
    exclude_columns = set(id_columns)
    
    # Window columns
    window_cols = [
        "pre_window_start", "pre_window_end",
        "post_window_start", "post_window_end"
    ]
    exclude_columns.update(window_cols)
    
    # Coverage flags
    coverage_flags = ["has_full_pre_window", "has_full_post_window"]
    exclude_columns.update(coverage_flags)
    
    # Target-related columns
    target_cols = [
        "stock_return_30d", "spy_return_30d",
        "excess_return_30d", "label_outperform_30d"
    ]
    exclude_columns.update(target_cols)
    
    # Additional metadata columns to exclude
    metadata_cols = ["quarter", "year"]
    exclude_columns.update(metadata_cols)
    
    # Build feature list
    feature_columns = []
    
    # Add all fundamental features (f_*)
    fundamental_features = [col for col in df.columns if col.startswith("f_")]
    feature_columns.extend(fundamental_features)
    
    # Add all market features
    market_features = [
        "stock_momentum_1m",
        "stock_momentum_3m",
        "stock_momentum_6m",
        "pre_volatility_30d",
        "pre_avg_volume_30d",
        "spy_pre_return_30d",
    ]
    
    for col in market_features:
        if col in df.columns and col not in feature_columns:
            feature_columns.append(col)
    
    # Sort for consistency
    feature_columns = sorted(feature_columns)
    
    # Create spec
    spec = DatasetSpec(
        id_columns=id_columns,
        feature_columns=feature_columns,
        target_regression="excess_return_30d",
        target_classification="label_outperform_30d",
    )
    
    logger.info(f"  Dataset specification built:")
    logger.info(f"    Identifiers: {len(spec.id_columns)} columns")
    logger.info(f"    Features: {len(spec.feature_columns)} columns")
    logger.info(f"    Regression target: {spec.target_regression}")
    logger.info(f"    Classification target: {spec.target_classification}")
    logger.info(f"  Example features: {spec.feature_columns[:5]}")
    
    return spec


def clean_and_filter_rows(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """
    Apply basic row-level cleaning:

    - Drop rows with missing regression target.
    - Optionally drop rows with missing classification label.
    - Keep only rows with has_full_post_window == True (to ensure target quality).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with all columns.
    spec : DatasetSpec
        Dataset specification.
    
    Returns
    -------
    pd.DataFrame
        Filtered copy of the dataframe.
    """
    logger.info("Cleaning and filtering rows...")
    
    initial_count = len(df)
    logger.info(f"  Initial row count: {initial_count:,}")
    
    # Work on a copy
    df_clean = df.copy()
    
    # Require regression target not NaN
    if spec.target_regression in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean[spec.target_regression].notna()]
        dropped = before - len(df_clean)
        if dropped > 0:
            logger.info(f"  Dropped {dropped:,} rows with missing {spec.target_regression}")
    
    # Require classification target not NaN (if present)
    if spec.target_classification in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean[spec.target_classification].notna()]
        dropped = before - len(df_clean)
        if dropped > 0:
            logger.info(f"  Dropped {dropped:,} rows with missing {spec.target_classification}")
    
    # Require full post-window coverage (to ensure target quality)
    if "has_full_post_window" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean["has_full_post_window"] == True]
        dropped = before - len(df_clean)
        if dropped > 0:
            logger.info(f"  Dropped {dropped:,} rows without full post-window coverage")
    
    final_count = len(df_clean)
    kept_pct = 100 * final_count / initial_count if initial_count > 0 else 0
    
    logger.info(f"  Final row count: {final_count:,} ({kept_pct:.1f}% kept)")
    
    return df_clean


def train_val_test_split_time_based(
    df: pd.DataFrame,
    spec: DatasetSpec,
    settings: Settings,
    test_split_date: str = "2020-01-01",
    val_fraction_within_train: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    """
    Split the dataset into train/validation/test using time-based splits.

    - Test set: all rows with earnings_date >= test_split_date.
    - Train+val: earnings_date < test_split_date.
    - Within pre-2020 data, use the last part as validation based on time.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    spec : DatasetSpec
        Dataset specification.
    settings : Settings
        Project settings.
    test_split_date : str, optional
        Date cutoff for test set (default: "2020-01-01").
    val_fraction_within_train : float, optional
        Fraction of pre-test data to use for validation (default: 0.2).
    
    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', 'test' mapping to dataframes.
    """
    logger.info("Performing time-based train/val/test split...")
    logger.info(f"  Test split date: {test_split_date}")
    logger.info(f"  Validation fraction (within pre-test): {val_fraction_within_train:.1%}")
    
    date_col = settings.EARNINGS_DATE_COLUMN
    
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date ascending
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Split into test and train+val
    test_cutoff = pd.Timestamp(test_split_date)
    
    df_test = df[df[date_col] >= test_cutoff].copy()
    df_train_val = df[df[date_col] < test_cutoff].copy()
    
    logger.info(f"  Pre-test data (train+val): {len(df_train_val):,} rows")
    logger.info(f"  Test data: {len(df_test):,} rows")
    
    # Within train+val, split into train and val based on time
    n_train_val = len(df_train_val)
    n_val = int(np.floor(val_fraction_within_train * n_train_val))
    n_train = n_train_val - n_val
    
    df_train = df_train_val.iloc[:n_train].copy()
    df_val = df_train_val.iloc[n_train:].copy()
    
    logger.info(f"\n  Final split:")
    logger.info(f"    Train: {len(df_train):,} rows")
    logger.info(f"    Val:   {len(df_val):,} rows")
    logger.info(f"    Test:  {len(df_test):,} rows")
    
    if len(df_train) > 0 and len(df_val) > 0:
        train_max_date = df_train[date_col].max()
        val_min_date = df_val[date_col].min()
        val_max_date = df_val[date_col].max()
        test_min_date = df_test[date_col].min() if len(df_test) > 0 else None
        
        logger.info(f"\n  Date ranges:")
        logger.info(f"    Train: up to {train_max_date.date()}")
        logger.info(f"    Val:   {val_min_date.date()} to {val_max_date.date()}")
        if test_min_date:
            logger.info(f"    Test:  from {test_min_date.date()}")
    
    return {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }


def scale_and_package_matrices(
    splits: Dict[str, pd.DataFrame],
    spec: DatasetSpec,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], StandardScaler]:
    """
    Build X/y matrices for each split and apply standard scaling to features.

    Scaling is fit only on the training data, then applied to val and test.

    Parameters
    ----------
    splits : dict
        Dict with keys 'train', 'val', 'test' and dataframe values.
    spec : DatasetSpec
        Dataset specification with feature and target columns.

    Returns
    -------
    X_splits : dict
        Mapping 'train'/'val'/'test' -> scaled feature matrix (2D np.ndarray).
    y_splits : dict
        Mapping 'train'/'val'/'test' -> regression target array (1D np.ndarray).
    scaler : StandardScaler
        Fitted scaler object.
    """
    logger.info("Building and scaling feature matrices...")
    
    X_splits = {}
    y_splits = {}
    
    # Extract raw matrices
    for split_name in ["train", "val", "test"]:
        df_split = splits[split_name]
        
        # Features
        X = df_split[spec.feature_columns].to_numpy(dtype=float)
        
        # Target (regression)
        y = df_split[spec.target_regression].to_numpy(dtype=float)
        
        X_splits[split_name] = X
        y_splits[split_name] = y
        
        logger.info(f"  {split_name:5s}: X shape {X.shape}, y shape {y.shape}")
    
    # Fit scaler on training data only
    logger.info("\n  Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_splits["train"] = scaler.fit_transform(X_splits["train"])
    
    logger.info(f"    Scaler fitted: mean shape {scaler.mean_.shape}, scale shape {scaler.scale_.shape}")
    
    # Transform validation and test
    X_splits["val"] = scaler.transform(X_splits["val"])
    X_splits["test"] = scaler.transform(X_splits["test"])
    
    logger.info("  ✓ Scaling applied to all splits")
    
    # Verify scaling on train
    train_mean = X_splits["train"].mean(axis=0)
    train_std = X_splits["train"].std(axis=0, ddof=0)
    
    logger.info(f"\n  Training data after scaling:")
    logger.info(f"    Mean (max abs): {np.abs(train_mean).max():.6f}")
    logger.info(f"    Std (mean): {train_std.mean():.6f}")
    
    return X_splits, y_splits, scaler


def save_preprocessed_data(
    splits: Dict[str, pd.DataFrame],
    X_splits: Dict[str, np.ndarray],
    y_splits: Dict[str, np.ndarray],
    scaler: StandardScaler,
    spec: DatasetSpec,
    output_dir: Path,
) -> None:
    """
    Save cleaned dataframes, scaled matrices, scaler, and metadata.

    Artifacts:
    - cleaned_{split}.parquet    (full dataframes with features+targets)
    - X_{split}.npy, y_{split}.npy
    - scaler.joblib
    - dataset_spec.json
    - split_summary.json
    
    Parameters
    ----------
    splits : dict
        Dictionary of dataframes for each split.
    X_splits : dict
        Dictionary of scaled feature matrices.
    y_splits : dict
        Dictionary of target arrays.
    scaler : StandardScaler
        Fitted scaler object.
    spec : DatasetSpec
        Dataset specification.
    output_dir : Path
        Output directory for all artifacts.
    """
    import joblib
    
    logger.info(f"\nSaving preprocessed data to {output_dir}...")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned dataframes
    for split_name, df_split in splits.items():
        parquet_path = output_dir / f"cleaned_{split_name}.parquet"
        df_split.to_parquet(parquet_path, index=False)
        logger.info(f"  ✓ Saved {parquet_path.name} ({len(df_split):,} rows)")
    
    # Save scaled matrices
    for split_name in ["train", "val", "test"]:
        X_path = output_dir / f"X_{split_name}.npy"
        y_path = output_dir / f"y_{split_name}.npy"
        
        np.save(X_path, X_splits[split_name])
        np.save(y_path, y_splits[split_name])
        
        logger.info(f"  ✓ Saved {X_path.name} {X_splits[split_name].shape}")
        logger.info(f"  ✓ Saved {y_path.name} {y_splits[split_name].shape}")
    
    # Save scaler
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"  ✓ Saved {scaler_path.name}")
    
    # Save dataset specification
    spec_dict = {
        "id_columns": spec.id_columns,
        "feature_columns": spec.feature_columns,
        "target_regression": spec.target_regression,
        "target_classification": spec.target_classification,
        "n_features": len(spec.feature_columns),
    }
    
    spec_path = output_dir / "dataset_spec.json"
    with spec_path.open("w", encoding="utf-8") as f:
        json.dump(spec_dict, f, indent=2)
    logger.info(f"  ✓ Saved {spec_path.name}")
    
    # Save split summary
    summary = {
        "train": {
            "n_rows": int(len(splits["train"])),
            "n_features": int(X_splits["train"].shape[1]),
            "target_mean": float(y_splits["train"].mean()),
            "target_std": float(y_splits["train"].std()),
        },
        "val": {
            "n_rows": int(len(splits["val"])),
            "n_features": int(X_splits["val"].shape[1]),
            "target_mean": float(y_splits["val"].mean()),
            "target_std": float(y_splits["val"].std()),
        },
        "test": {
            "n_rows": int(len(splits["test"])),
            "n_features": int(X_splits["test"].shape[1]),
            "target_mean": float(y_splits["test"].mean()),
            "target_std": float(y_splits["test"].std()),
        },
    }
    
    summary_path = output_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  ✓ Saved {summary_path.name}")
    
    logger.info("\n  All artifacts saved successfully!")
