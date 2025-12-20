"""
Rolling-window out-of-sample evaluation.

This module implements time-based rolling train/test splits to assess
temporal robustness of model performance and H₀ conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.config import Settings
from src.metrics.regression import regression_metrics
from src.models.baselines import estimate_capm_betas, predict_capm_baseline


logger = logging.getLogger(__name__)


@dataclass
class RollingFoldResult:
    """
    Container for metrics of a single rolling window fold.
    
    Attributes
    ----------
    fold_id : int
        Fold identifier (1-indexed).
    train_start : str
        Start date of training window.
    train_end : str
        End date of training window.
    test_start : str
        Start date of test window.
    test_end : str
        End date of test window.
    metrics : dict
        Mapping from model_name to metrics dict (train_mae, test_mae, etc.).
    """
    
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: Dict[str, Dict[str, float]]


def build_full_cleaned_panel(settings: Settings) -> pd.DataFrame:
    """
    Concatenate cleaned train/val/test dataframes from Step 10 into a single
    time-ordered panel.

    Parameters
    ----------
    settings : Settings
        Project settings.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe sorted by earnings date, with all features
        and the regression target 'excess_return_30d'.
    """
    logger.info("Building full cleaned panel from Step 10...")
    
    step10_dir = settings.RESULTS_DIR / "step_10"
    
    # Load the three cleaned parquet files
    df_train = pd.read_parquet(step10_dir / "cleaned_train.parquet")
    df_val = pd.read_parquet(step10_dir / "cleaned_val.parquet")
    df_test = pd.read_parquet(step10_dir / "cleaned_test.parquet")
    
    logger.info(f"  Train: {len(df_train)} rows")
    logger.info(f"  Val: {len(df_val)} rows")
    logger.info(f"  Test: {len(df_test)} rows")
    
    # Concatenate
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    # Convert earnings date to datetime if not already
    date_col = settings.EARNINGS_DATE_COLUMN
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date ascending
    df = df.sort_values(date_col).reset_index(drop=True)
    
    logger.info(f"  Total panel: {len(df)} rows")
    logger.info(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")
    
    return df


def generate_time_slices(
    df: pd.DataFrame,
    settings: Settings,
    min_train_years: int = 5,
) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Generate rolling train/test index splits based on calendar years.

    Strategy:
    - Use year extracted from earnings date column.
    - For each year >= (min_train_years + first_year), define:
        * train_idx: all rows with year < current_year
        * test_idx: all rows with year == current_year
    - Only keep splits where both train and test sets are non-empty.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned panel sorted by date.
    settings : Settings
        Project settings.
    min_train_years : int, default=5
        Minimum number of years to use for initial training window.

    Returns
    -------
    list of (train_idx, test_idx)
        List of index pairs defining rolling folds.
    """
    logger.info("Generating time-based rolling splits...")
    
    date_col = settings.EARNINGS_DATE_COLUMN
    
    # Extract year
    df['_year'] = df[date_col].dt.year
    
    years = sorted(df['_year'].unique())
    first_year = years[0]
    last_year = years[-1]
    
    logger.info(f"  Years available: {first_year} to {last_year}")
    logger.info(f"  Min train years: {min_train_years}")
    
    splits = []
    
    # Start testing from (first_year + min_train_years)
    for test_year in range(first_year + min_train_years, last_year + 1):
        # Train on all years before test_year
        train_idx = df[df['_year'] < test_year].index
        test_idx = df[df['_year'] == test_year].index
        
        # Only keep if both sets are non-empty
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
            logger.info(
                f"  Fold {len(splits)}: Train {first_year}-{test_year-1} "
                f"({len(train_idx)} obs) → Test {test_year} ({len(test_idx)} obs)"
            )
    
    # Clean up temporary column
    df.drop(columns=['_year'], inplace=True)
    
    logger.info(f"  Generated {len(splits)} rolling folds")
    
    return splits


def train_and_evaluate_on_fold(
    df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    settings: Settings,
    fold_id: int,
) -> RollingFoldResult:
    """
    Train all selected models on the training subset and evaluate them on the
    test subset defined by the provided indices.

    Models:
    - baseline_mean
    - baseline_capm
    - ridge
    - random_forest
    - xgb_best (HistGB fallback)

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned panel.
    train_idx : pd.Index
        Indices for training set.
    test_idx : pd.Index
        Indices for test set.
    settings : Settings
        Project settings.
    fold_id : int
        Fold identifier.

    Returns
    -------
    RollingFoldResult
        Result object containing metrics for all models on this fold.
    """
    logger.info(f"Training and evaluating fold {fold_id}...")
    
    # Subset dataframes
    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()
    
    # Get date range
    date_col = settings.EARNINGS_DATE_COLUMN
    train_start = df_train[date_col].min().strftime('%Y-%m-%d')
    train_end = df_train[date_col].max().strftime('%Y-%m-%d')
    test_start = df_test[date_col].min().strftime('%Y-%m-%d')
    test_end = df_test[date_col].max().strftime('%Y-%m-%d')
    
    # Load feature columns from dataset_spec
    step10_dir = settings.RESULTS_DIR / "step_10"
    with (step10_dir / "dataset_spec.json").open("r") as f:
        dataset_spec = json.load(f)
    
    feature_cols = dataset_spec["feature_columns"]
    target_col = "excess_return_30d"
    
    # Build X / y
    X_train_raw = df_train[feature_cols].to_numpy()
    X_test_raw = df_test[feature_cols].to_numpy()
    y_train = df_train[target_col].to_numpy()
    y_test = df_test[target_col].to_numpy()
    
    # Imputation + Scaling (fit on train only)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_test = scaler.transform(X_test_imp)
    
    # Dictionary to store metrics
    metrics = {}
    
    # 1. Baseline Mean
    logger.info("  Training baseline_mean...")
    baseline_mean = DummyRegressor(strategy="mean")
    baseline_mean.fit(np.zeros((len(y_train), 1)), y_train)
    
    y_pred_train_mean = baseline_mean.predict(np.zeros((len(y_train), 1)))
    y_pred_test_mean = baseline_mean.predict(np.zeros((len(y_test), 1)))
    
    metrics["baseline_mean"] = {
        **regression_metrics(y_train, y_pred_train_mean, prefix="train_"),
        **regression_metrics(y_test, y_pred_test_mean, prefix="test_"),
    }
    
    # 2. CAPM Baseline
    logger.info("  Training baseline_capm...")
    capm_betas = estimate_capm_betas(df_train, settings)
    
    y_pred_train_capm = predict_capm_baseline(df_train, capm_betas, settings)
    y_pred_test_capm = predict_capm_baseline(df_test, capm_betas, settings)
    
    metrics["baseline_capm"] = {
        **regression_metrics(y_train, y_pred_train_capm, prefix="train_"),
        **regression_metrics(y_test, y_pred_test_capm, prefix="test_"),
    }
    
    # 3. Ridge
    logger.info("  Training ridge...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    
    y_pred_train_ridge = ridge.predict(X_train)
    y_pred_test_ridge = ridge.predict(X_test)
    
    metrics["ridge"] = {
        **regression_metrics(y_train, y_pred_train_ridge, prefix="train_"),
        **regression_metrics(y_test, y_pred_test_ridge, prefix="test_"),
    }
    
    # 4. Random Forest
    logger.info("  Training random_forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    
    y_pred_train_rf = rf.predict(X_train)
    y_pred_test_rf = rf.predict(X_test)
    
    metrics["random_forest"] = {
        **regression_metrics(y_train, y_pred_train_rf, prefix="train_"),
        **regression_metrics(y_test, y_pred_test_rf, prefix="test_"),
    }
    
    # 5. XGB Best (HistGB fallback)
    logger.info("  Training xgb_best (HistGB)...")
    xgb = HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    
    y_pred_train_xgb = xgb.predict(X_train)
    y_pred_test_xgb = xgb.predict(X_test)
    
    metrics["xgb_best"] = {
        **regression_metrics(y_train, y_pred_train_xgb, prefix="train_"),
        **regression_metrics(y_test, y_pred_test_xgb, prefix="test_"),
    }
    
    logger.info(f"  ✓ Fold {fold_id} complete")
    
    return RollingFoldResult(
        fold_id=fold_id,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        metrics=metrics,
    )


def run_rolling_window_evaluation(
    settings: Settings,
    min_train_years: int = 5,
) -> List[RollingFoldResult]:
    """
    Run rolling-window OOS evaluation across all time slices.

    Parameters
    ----------
    settings : Settings
        Project settings.
    min_train_years : int, default=5
        Minimum number of years for initial training window.

    Returns
    -------
    list of RollingFoldResult
        One entry per rolling fold.
    """
    logger.info("=" * 70)
    logger.info("ROLLING-WINDOW OUT-OF-SAMPLE EVALUATION")
    logger.info("=" * 70)
    
    # Build full panel
    df = build_full_cleaned_panel(settings)
    
    # Generate time slices
    splits = generate_time_slices(df, settings, min_train_years=min_train_years)
    
    if len(splits) == 0:
        raise ValueError("No valid rolling splits generated. Check data and min_train_years.")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"TRAINING MODELS ON {len(splits)} ROLLING FOLDS")
    logger.info("=" * 70)
    
    # Train and evaluate on each fold
    results: List[RollingFoldResult] = []
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        logger.info("")
        logger.info(f"Fold {fold_id} / {len(splits)}")
        logger.info("-" * 70)
        
        fold_result = train_and_evaluate_on_fold(
            df, train_idx, test_idx, settings, fold_id
        )
        results.append(fold_result)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("ROLLING-WINDOW EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total folds: {len(results)}")
    
    return results
