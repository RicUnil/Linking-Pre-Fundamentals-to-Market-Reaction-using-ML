"""
Unit tests for model comparison (Step 14).

This module tests the model loading and comparison functionality.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import json


def test_model_comparison_loads_all_models():
    """Test that all 10 models can be loaded without error."""
    from src.config import Settings
    from src.step_14_compare_models import load_all_models
    
    settings = Settings()
    
    # Check that required directories exist
    step11_dir = settings.RESULTS_DIR / "step_11"
    step12_dir = settings.RESULTS_DIR / "step_12"
    step13_dir = settings.RESULTS_DIR / "step_13"
    
    if not (step11_dir.exists() and step12_dir.exists() and step13_dir.exists()):
        pytest.skip("Required model directories not found (Steps 11-13 not run)")
    
    # Load all models
    models = load_all_models(settings)
    
    # Check that we have 10 models
    assert len(models) == 10
    
    # Check that all expected models are present
    expected_models = [
        "baseline_mean", "baseline_capm", "linear_ols", "ridge", "lasso",
        "random_forest", "gradient_boosting", "hist_gradient_boosting",
        "xgb_baseline", "xgb_tuned"
    ]
    
    for model_name in expected_models:
        assert model_name in models, f"Model {model_name} not found"


def test_predict_capm_baseline_shape():
    """Test that CAPM baseline predictions have correct shape."""
    from src.step_14_compare_models import predict_capm_baseline
    
    # Create synthetic betas (ticker as index)
    betas_df = pd.DataFrame({
        'beta': [1.2, 0.8, 1.0],
        'alpha': [0.01, -0.005, 0.0]
    }, index=['AAPL', 'GOOGL', 'MSFT'])
    betas_df.index.name = 'ticker'
    
    # Create synthetic data
    df_clean = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AAPL'],
        'spy_pre_return_30d': [0.05, -0.02, 0.03, 0.01, -0.01]
    })
    
    y_indices = np.array([0, 1, 2, 3, 4])
    global_mean = 0.02
    
    predictions = predict_capm_baseline(betas_df, df_clean, y_indices, global_mean)
    
    # Check shape
    assert predictions.shape == (5,)
    
    # Check that predictions are not all the same (some should use beta, some mean)
    assert len(np.unique(predictions)) > 1


def test_predict_capm_baseline_fallback_to_mean():
    """Test that CAPM baseline falls back to mean for unknown tickers."""
    from src.step_14_compare_models import predict_capm_baseline
    
    # Create synthetic betas (only AAPL, ticker as index)
    betas_df = pd.DataFrame({
        'beta': [1.2],
        'alpha': [0.01]
    }, index=['AAPL'])
    betas_df.index.name = 'ticker'
    
    # Create synthetic data with unknown ticker
    df_clean = pd.DataFrame({
        'ticker': ['UNKNOWN'],
        'spy_pre_return_30d': [0.05]
    })
    
    y_indices = np.array([0])
    global_mean = 0.02
    
    predictions = predict_capm_baseline(betas_df, df_clean, y_indices, global_mean)
    
    # Should fall back to global mean
    assert predictions[0] == global_mean


def test_model_comparison_output_files_exist():
    """Test that Step 14 creates all expected output files."""
    from src.config import Settings
    
    settings = Settings()
    output_dir = settings.RESULTS_DIR / "step_14"
    
    if not output_dir.exists():
        pytest.skip("Step 14 not run yet")
    
    # Check for expected files
    expected_files = [
        "model_comparison.csv",
        "model_comparison.json",
        "ranked_models.json",
        "step_14_completed.txt"
    ]
    
    for filename in expected_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"Expected file {filename} not found"


def test_model_comparison_json_structure():
    """Test that model_comparison.json has correct structure."""
    from src.config import Settings
    
    settings = Settings()
    json_path = settings.RESULTS_DIR / "step_14" / "model_comparison.json"
    
    if not json_path.exists():
        pytest.skip("Step 14 not run yet")
    
    with json_path.open("r") as f:
        data = json.load(f)
    
    # Check that we have 10 models
    assert len(data) == 10
    
    # Check that each model has required metrics
    required_keys = [
        'category', 'train_mae', 'train_rmse', 'train_r2',
        'val_mae', 'val_rmse', 'val_r2'
    ]
    
    for model_name, metrics in data.items():
        for key in required_keys:
            assert key in metrics, f"Model {model_name} missing key {key}"
        
        # Check that metrics are numeric
        assert isinstance(metrics['train_mae'], (int, float))
        assert isinstance(metrics['val_mae'], (int, float))
        
        # Check that metrics are not NaN
        assert not np.isnan(metrics['train_mae'])
        assert not np.isnan(metrics['val_mae'])


def test_ranked_models_json_structure():
    """Test that ranked_models.json has correct structure and sorting."""
    from src.config import Settings
    
    settings = Settings()
    json_path = settings.RESULTS_DIR / "step_14" / "ranked_models.json"
    
    if not json_path.exists():
        pytest.skip("Step 14 not run yet")
    
    with json_path.open("r") as f:
        ranked = json.load(f)
    
    # Check that we have 10 models
    assert len(ranked) == 10
    
    # Check that ranks are sequential
    ranks = [m['rank'] for m in ranked]
    assert ranks == list(range(1, 11))
    
    # Check that models are sorted by val_mae (ascending)
    val_maes = [m['val_mae'] for m in ranked]
    assert val_maes == sorted(val_maes)
    
    # Check structure of each entry
    required_keys = [
        'rank', 'model_name', 'category',
        'val_mae', 'val_rmse', 'val_r2',
        'train_mae', 'train_rmse', 'train_r2'
    ]
    
    for model in ranked:
        for key in required_keys:
            assert key in model, f"Ranked model missing key {key}"


def test_model_comparison_csv_structure():
    """Test that model_comparison.csv has correct structure."""
    from src.config import Settings
    
    settings = Settings()
    csv_path = settings.RESULTS_DIR / "step_14" / "model_comparison.csv"
    
    if not csv_path.exists():
        pytest.skip("Step 14 not run yet")
    
    df = pd.read_csv(csv_path)
    
    # Check shape
    assert len(df) == 10
    
    # Check columns
    expected_columns = [
        'model_name', 'category',
        'train_mae', 'train_rmse', 'train_r2',
        'val_mae', 'val_rmse', 'val_r2'
    ]
    
    for col in expected_columns:
        assert col in df.columns, f"Column {col} not found in CSV"
    
    # Check that no NaNs in metrics
    metric_columns = [
        'train_mae', 'train_rmse', 'train_r2',
        'val_mae', 'val_rmse', 'val_r2'
    ]
    
    for col in metric_columns:
        assert not df[col].isna().any(), f"Column {col} contains NaN values"


def test_model_comparison_categories():
    """Test that models are assigned to correct categories."""
    from src.config import Settings
    
    settings = Settings()
    csv_path = settings.RESULTS_DIR / "step_14" / "model_comparison.csv"
    
    if not csv_path.exists():
        pytest.skip("Step 14 not run yet")
    
    df = pd.read_csv(csv_path)
    
    # Check category assignments
    expected_categories = {
        'baseline_mean': 'Baseline',
        'baseline_capm': 'Baseline',
        'linear_ols': 'Linear',
        'ridge': 'Linear',
        'lasso': 'Linear',
        'random_forest': 'Tree',
        'gradient_boosting': 'Tree',
        'hist_gradient_boosting': 'Tree',
        'xgb_baseline': 'Boosting',
        'xgb_tuned': 'Boosting',
    }
    
    for _, row in df.iterrows():
        model_name = row['model_name']
        if model_name in expected_categories:
            assert row['category'] == expected_categories[model_name]
