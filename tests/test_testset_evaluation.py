"""
Unit tests for test-set evaluation (Step 15).

This module tests the test-set evaluation functionality.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import json


def test_regression_metrics_on_constant_prediction():
    """Test that regression metrics work on constant predictions."""
    from src.metrics.regression import regression_metrics
    
    y_true = np.array([0.0, 0.1, -0.1, 0.05])
    y_pred = np.zeros_like(y_true)

    m = regression_metrics(y_true, y_pred, prefix="test_")
    
    assert "test_mae" in m
    assert "test_rmse" in m
    assert "test_r2" in m
    
    # Check that metrics are numeric
    assert isinstance(m["test_mae"], (int, float))
    assert isinstance(m["test_rmse"], (int, float))
    assert isinstance(m["test_r2"], (int, float))


def test_regression_metrics_perfect_prediction():
    """Test that regression metrics work on perfect predictions."""
    from src.metrics.regression import regression_metrics
    
    y_true = np.array([0.0, 0.1, -0.1, 0.05, 0.02])
    y_pred = y_true.copy()

    m = regression_metrics(y_true, y_pred, prefix="test_")
    
    # Perfect predictions should have MAE=0, RMSE=0, R²=1
    assert m["test_mae"] == 0.0
    assert m["test_rmse"] == 0.0
    assert m["test_r2"] == 1.0


def test_regression_metrics_shape_mismatch():
    """Test that regression metrics raise error on shape mismatch."""
    from src.metrics.regression import regression_metrics
    
    y_true = np.array([0.0, 0.1, -0.1])
    y_pred = np.array([0.0, 0.1])  # Different length
    
    with pytest.raises(ValueError):
        regression_metrics(y_true, y_pred, prefix="test_")


def test_load_selected_models():
    """Test that selected models can be loaded."""
    from src.config import Settings
    from src.step_15_evaluate_on_test import load_selected_models
    
    settings = Settings()
    
    # Check that required directories exist
    step11_dir = settings.RESULTS_DIR / "step_11"
    step12_dir = settings.RESULTS_DIR / "step_12"
    step13_dir = settings.RESULTS_DIR / "step_13"
    
    if not (step11_dir.exists() and step12_dir.exists() and step13_dir.exists()):
        pytest.skip("Required model directories not found (Steps 11-13 not run)")
    
    # Load models
    models = load_selected_models(settings)
    
    # Check that we have 5 models
    assert len(models) == 5
    
    # Check that all expected models are present
    expected_models = [
        "baseline_mean", "capm_betas", "ridge",
        "random_forest", "xgb_best"
    ]
    
    for model_name in expected_models:
        assert model_name in models, f"Model {model_name} not found"


def test_test_metrics_output_files_exist():
    """Test that Step 15 creates all expected output files."""
    from src.config import Settings
    
    settings = Settings()
    output_dir = settings.RESULTS_DIR / "step_15"
    
    if not output_dir.exists():
        pytest.skip("Step 15 not run yet")
    
    # Check for expected files
    expected_files = [
        "test_metrics.json",
        "test_metrics.csv",
        "ridge_test_residuals.parquet",
        "step_15_completed.txt"
    ]
    
    for filename in expected_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"Expected file {filename} not found"


def test_test_metrics_json_structure():
    """Test that test_metrics.json has correct structure."""
    from src.config import Settings
    
    settings = Settings()
    json_path = settings.RESULTS_DIR / "step_15" / "test_metrics.json"
    
    if not json_path.exists():
        pytest.skip("Step 15 not run yet")
    
    with json_path.open("r") as f:
        data = json.load(f)
    
    # Check that we have 5 models
    assert len(data) == 5
    
    # Check that each model has required metrics
    required_keys = ['test_mae', 'test_rmse', 'test_r2']
    
    expected_models = [
        "baseline_mean", "baseline_capm", "ridge",
        "random_forest", "xgb_best"
    ]
    
    for model_name in expected_models:
        assert model_name in data, f"Model {model_name} not found in test metrics"
        
        metrics = data[model_name]
        for key in required_keys:
            assert key in metrics, f"Model {model_name} missing key {key}"
        
        # Check that metrics are numeric
        assert isinstance(metrics['test_mae'], (int, float))
        assert isinstance(metrics['test_rmse'], (int, float))
        assert isinstance(metrics['test_r2'], (int, float))
        
        # Check that metrics are not NaN
        assert not np.isnan(metrics['test_mae'])
        assert not np.isnan(metrics['test_rmse'])
        assert not np.isnan(metrics['test_r2'])


def test_test_metrics_csv_structure():
    """Test that test_metrics.csv has correct structure."""
    from src.config import Settings
    
    settings = Settings()
    csv_path = settings.RESULTS_DIR / "step_15" / "test_metrics.csv"
    
    if not csv_path.exists():
        pytest.skip("Step 15 not run yet")
    
    df = pd.read_csv(csv_path)
    
    # Check shape
    assert len(df) == 5
    
    # Check columns
    expected_columns = ['model_name', 'test_mae', 'test_rmse', 'test_r2']
    
    for col in expected_columns:
        assert col in df.columns, f"Column {col} not found in CSV"
    
    # Check that no NaNs in metrics
    metric_columns = ['test_mae', 'test_rmse', 'test_r2']
    
    for col in metric_columns:
        assert not df[col].isna().any(), f"Column {col} contains NaN values"
    
    # Check that CSV is sorted by test_mae
    assert df['test_mae'].is_monotonic_increasing, "CSV should be sorted by test_mae"


def test_ridge_residuals_structure():
    """Test that ridge_test_residuals.parquet has correct structure."""
    from src.config import Settings
    
    settings = Settings()
    residuals_path = settings.RESULTS_DIR / "step_15" / "ridge_test_residuals.parquet"
    
    if not residuals_path.exists():
        pytest.skip("Step 15 not run yet")
    
    df = pd.read_parquet(residuals_path)
    
    # Check columns
    ticker_col = settings.EARNINGS_TICKER_COLUMN
    date_col = settings.EARNINGS_DATE_COLUMN
    
    expected_columns = [
        ticker_col, date_col,
        'y_true', 'y_pred_ridge', 'residual_ridge'
    ]
    
    for col in expected_columns:
        assert col in df.columns, f"Column {col} not found in residuals"
    
    # Check that residuals are computed correctly
    computed_residuals = df['y_true'] - df['y_pred_ridge']
    np.testing.assert_array_almost_equal(
        df['residual_ridge'].values,
        computed_residuals.values,
        decimal=10,
        err_msg="Residuals not computed correctly"
    )
    
    # Check that no NaNs in predictions and residuals
    assert not df['y_pred_ridge'].isna().any(), "y_pred_ridge contains NaN values"
    assert not df['residual_ridge'].isna().any(), "residual_ridge contains NaN values"


def test_test_metrics_consistency_with_validation():
    """Test that test metrics are in similar range to validation metrics."""
    from src.config import Settings
    import json
    
    settings = Settings()
    
    test_metrics_path = settings.RESULTS_DIR / "step_15" / "test_metrics.json"
    val_metrics_path = settings.RESULTS_DIR / "step_14" / "model_comparison.json"
    
    if not (test_metrics_path.exists() and val_metrics_path.exists()):
        pytest.skip("Step 14 or Step 15 not run yet")
    
    with test_metrics_path.open("r") as f:
        test_metrics = json.load(f)
    
    with val_metrics_path.open("r") as f:
        val_metrics = json.load(f)
    
    # Check Ridge metrics consistency
    if "ridge" in test_metrics and "ridge" in val_metrics:
        test_mae = test_metrics["ridge"]["test_mae"]
        val_mae = val_metrics["ridge"]["val_mae"]
        
        # Test and validation MAE should be in similar range (within 50%)
        # This is a sanity check, not a strict requirement
        ratio = test_mae / val_mae if val_mae > 0 else 1.0
        assert 0.5 < ratio < 2.0, f"Test MAE ({test_mae:.6f}) and Val MAE ({val_mae:.6f}) differ too much"
