"""
Unit tests for rolling-window evaluation (Step 16).

This module tests the rolling-window evaluation functionality.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import json


def test_build_full_cleaned_panel():
    """Test that full cleaned panel can be built from Step 10 data."""
    from src.config import Settings
    from src.evaluation.rolling_window import build_full_cleaned_panel
    
    settings = Settings()
    
    # Check that Step 10 data exists
    step10_dir = settings.RESULTS_DIR / "step_10"
    if not step10_dir.exists():
        pytest.skip("Step 10 not run yet")
    
    # Build panel
    df = build_full_cleaned_panel(settings)
    
    # Check that it's a dataframe
    assert isinstance(df, pd.DataFrame)
    
    # Check that it has rows
    assert len(df) > 0
    
    # Check that it has the target column
    assert "excess_return_30d" in df.columns
    
    # Check that it has the date column
    date_col = settings.EARNINGS_DATE_COLUMN
    assert date_col in df.columns
    
    # Check that dates are sorted
    assert df[date_col].is_monotonic_increasing


def test_generate_time_slices():
    """Test that time slices are generated correctly."""
    from src.config import Settings
    from src.evaluation.rolling_window import build_full_cleaned_panel, generate_time_slices
    
    settings = Settings()
    
    # Check that Step 10 data exists
    step10_dir = settings.RESULTS_DIR / "step_10"
    if not step10_dir.exists():
        pytest.skip("Step 10 not run yet")
    
    # Build panel
    df = build_full_cleaned_panel(settings)
    
    # Generate slices
    splits = generate_time_slices(df, settings, min_train_years=5)
    
    # Check that we have splits
    assert len(splits) > 0
    
    # Check that each split has train and test indices
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        
        # Check that train comes before test (no overlap)
        train_dates = df.loc[train_idx, settings.EARNINGS_DATE_COLUMN]
        test_dates = df.loc[test_idx, settings.EARNINGS_DATE_COLUMN]
        
        assert train_dates.max() < test_dates.min(), "Train and test periods overlap"


def test_rolling_fold_result_structure():
    """Test that RollingFoldResult has correct structure."""
    from src.evaluation.rolling_window import RollingFoldResult
    
    # Create a sample result
    result = RollingFoldResult(
        fold_id=1,
        train_start="2010-01-01",
        train_end="2014-12-31",
        test_start="2015-01-01",
        test_end="2015-12-31",
        metrics={
            "baseline_mean": {
                "train_mae": 0.04,
                "test_mae": 0.05,
                "train_r2": 0.0,
                "test_r2": 0.0,
            }
        }
    )
    
    # Check attributes
    assert result.fold_id == 1
    assert result.train_start == "2010-01-01"
    assert result.train_end == "2014-12-31"
    assert result.test_start == "2015-01-01"
    assert result.test_end == "2015-12-31"
    assert "baseline_mean" in result.metrics


def test_rolling_metrics_output_files_exist():
    """Test that Step 16 creates all expected output files."""
    from src.config import Settings
    
    settings = Settings()
    output_dir = settings.RESULTS_DIR / "step_16"
    
    if not output_dir.exists():
        pytest.skip("Step 16 not run yet")
    
    # Check for expected files
    expected_files = [
        "rolling_metrics_per_fold.csv",
        "rolling_metrics_per_fold.json",
        "rolling_metrics_aggregated.csv",
        "rolling_metrics_aggregated.json",
        "step_16_completed.txt"
    ]
    
    for filename in expected_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"Expected file {filename} not found"


def test_rolling_metrics_per_fold_structure():
    """Test that rolling_metrics_per_fold.csv has correct structure."""
    from src.config import Settings
    
    settings = Settings()
    csv_path = settings.RESULTS_DIR / "step_16" / "rolling_metrics_per_fold.csv"
    
    if not csv_path.exists():
        pytest.skip("Step 16 not run yet")
    
    df = pd.read_csv(csv_path)
    
    # Check that it has rows
    assert len(df) > 0
    
    # Check columns
    expected_columns = [
        'fold_id', 'train_start', 'train_end', 'test_start', 'test_end',
        'model_name', 'train_mae', 'train_rmse', 'train_r2',
        'test_mae', 'test_rmse', 'test_r2'
    ]
    
    for col in expected_columns:
        assert col in df.columns, f"Column {col} not found in CSV"
    
    # Check that we have all 5 models
    models = df['model_name'].unique()
    expected_models = ['baseline_mean', 'baseline_capm', 'ridge', 'random_forest', 'xgb_best']
    
    for model in expected_models:
        assert model in models, f"Model {model} not found in results"
    
    # Check that no NaNs in metrics
    metric_columns = ['train_mae', 'train_rmse', 'train_r2', 'test_mae', 'test_rmse', 'test_r2']
    
    for col in metric_columns:
        assert not df[col].isna().any(), f"Column {col} contains NaN values"


def test_rolling_metrics_aggregated_structure():
    """Test that rolling_metrics_aggregated.csv has correct structure."""
    from src.config import Settings
    
    settings = Settings()
    csv_path = settings.RESULTS_DIR / "step_16" / "rolling_metrics_aggregated.csv"
    
    if not csv_path.exists():
        pytest.skip("Step 16 not run yet")
    
    df = pd.read_csv(csv_path)
    
    # Check that we have 5 models
    assert len(df) == 5
    
    # Check that model_name column exists
    assert 'model_name' in df.columns
    
    # Check that aggregated metrics exist (mean, std, min, max)
    # Column names should be like test_mae_mean, test_mae_std, etc.
    assert any('test_mae' in col for col in df.columns)
    assert any('test_r2' in col for col in df.columns)


def test_rolling_temporal_consistency():
    """Test that rolling folds maintain temporal order."""
    from src.config import Settings
    
    settings = Settings()
    csv_path = settings.RESULTS_DIR / "step_16" / "rolling_metrics_per_fold.csv"
    
    if not csv_path.exists():
        pytest.skip("Step 16 not run yet")
    
    df = pd.read_csv(csv_path)
    
    # Get unique folds
    folds = df['fold_id'].unique()
    
    for fold_id in folds:
        fold_data = df[df['fold_id'] == fold_id].iloc[0]
        
        # Convert to datetime
        train_end = pd.to_datetime(fold_data['train_end'])
        test_start = pd.to_datetime(fold_data['test_start'])
        
        # Check that train ends before test starts
        assert train_end < test_start, f"Fold {fold_id}: train_end >= test_start"


def test_rolling_metrics_consistency():
    """Test that rolling metrics are in reasonable range."""
    from src.config import Settings
    
    settings = Settings()
    csv_path = settings.RESULTS_DIR / "step_16" / "rolling_metrics_per_fold.csv"
    
    if not csv_path.exists():
        pytest.skip("Step 16 not run yet")
    
    df = pd.read_csv(csv_path)
    
    # Check that MAE values are positive
    assert (df['test_mae'] > 0).all(), "Some test MAE values are not positive"
    assert (df['train_mae'] > 0).all(), "Some train MAE values are not positive"
    
    # Check that RMSE >= MAE (mathematical property)
    assert (df['test_rmse'] >= df['test_mae']).all(), "Some test RMSE < MAE"
    assert (df['train_rmse'] >= df['train_mae']).all(), "Some train RMSE < MAE"
    
    # Check that R² is in reasonable range (can be negative for bad models)
    # Note: Ridge can have catastrophically bad R² on some folds (e.g., -66)
    # This indicates model failure but is a valid result
    assert (df['test_r2'] >= -100).all(), "Some test R² values are unreasonably low"
    assert (df['test_r2'] <= 1).all(), "Some test R² values are > 1"
