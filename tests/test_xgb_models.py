"""
Unit tests for XGBoost regression models.

This module tests the XGBoost models (baseline and tuned) to ensure
correct behavior.
"""

import numpy as np
import pytest


def test_xgb_models_train_and_return_metrics():
    """Test that XGBoost models train and return proper metrics."""
    # Try to import xgboost, skip test if not available
    try:
        from src.models.xgb_models import train_xgb_models
    except ImportError:
        pytest.skip("xgboost is not installed")
    
    # Simple synthetic dataset with linear relationship
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(200, 6))
    true_coef = np.array([0.4, -0.3, 0.2, 0.0, 0.1, -0.05])
    y_train = X_train @ true_coef + 0.01 * rng.normal(size=200)
    
    X_val = rng.normal(size=(80, 6))
    y_val = X_val @ true_coef + 0.01 * rng.normal(size=80)
    
    models, metrics = train_xgb_models(X_train, y_train, X_val, y_val)
    
    # Check that both XGBoost models are present
    assert "xgb_baseline" in metrics
    assert "xgb_tuned" in metrics
    
    # Check that all metrics are present for each model
    for model_name in ["xgb_baseline", "xgb_tuned"]:
        assert "train_mae" in metrics[model_name]
        assert "train_rmse" in metrics[model_name]
        assert "train_r2" in metrics[model_name]
        assert "val_mae" in metrics[model_name]
        assert "val_rmse" in metrics[model_name]
        assert "val_r2" in metrics[model_name]
    
    # Check that models are fitted
    assert hasattr(models.baseline, 'feature_importances_')
    assert hasattr(models.tuned, 'feature_importances_')


def test_xgb_models_reasonable_performance():
    """Test that XGBoost models achieve reasonable performance on synthetic data."""
    try:
        from src.models.xgb_models import train_xgb_models
    except ImportError:
        pytest.skip("xgboost is not installed")
    
    # Create data with strong relationship
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(300, 5))
    true_coef = np.array([1.0, -0.5, 0.3, 0.0, 0.2])
    y_train = X_train @ true_coef + 0.05 * rng.normal(size=300)
    
    X_val = rng.normal(size=(100, 5))
    y_val = X_val @ true_coef + 0.05 * rng.normal(size=100)
    
    models, metrics = train_xgb_models(X_train, y_train, X_val, y_val)
    
    # XGBoost should achieve good R² on this synthetic data
    assert metrics["xgb_baseline"]["val_r2"] > 0.7
    assert metrics["xgb_tuned"]["val_r2"] > 0.7


def test_xgb_models_handle_more_samples_than_features():
    """Test that XGBoost models work with more samples than features."""
    try:
        from src.models.xgb_models import train_xgb_models
    except ImportError:
        pytest.skip("xgboost is not installed")
    
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(500, 3))
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + 0.1 * rng.normal(size=500)
    
    X_val = rng.normal(size=(100, 3))
    y_val = X_val[:, 0] + 0.5 * X_val[:, 1] + 0.1 * rng.normal(size=100)
    
    models, metrics = train_xgb_models(X_train, y_train, X_val, y_val)
    
    # Should complete without errors
    assert len(metrics) == 2
    assert all("val_mae" in m for m in metrics.values())


def test_xgb_models_feature_importances():
    """Test that XGBoost models compute feature importances."""
    try:
        from src.models.xgb_models import train_xgb_models
    except ImportError:
        pytest.skip("xgboost is not installed")
    
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(200, 5))
    # Make first feature very predictive
    y_train = 2.0 * X_train[:, 0] + 0.01 * rng.normal(size=200)
    
    X_val = rng.normal(size=(80, 5))
    y_val = 2.0 * X_val[:, 0] + 0.01 * rng.normal(size=80)
    
    models, metrics = train_xgb_models(X_train, y_train, X_val, y_val)
    
    # Check that feature importances exist
    for model in [models.baseline, models.tuned]:
        importances = model.feature_importances_
        assert len(importances) == 5
        # First feature should have highest importance
        assert importances[0] > 0.3


def test_xgb_models_predictions_shape():
    """Test that predictions have correct shape."""
    try:
        from src.models.xgb_models import train_xgb_models
    except ImportError:
        pytest.skip("xgboost is not installed")
    
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(100, 4))
    y_train = rng.normal(size=100)
    
    X_val = rng.normal(size=(50, 4))
    y_val = rng.normal(size=50)
    
    models, metrics = train_xgb_models(X_train, y_train, X_val, y_val)
    
    # Test predictions
    train_pred_baseline = models.baseline.predict(X_train)
    val_pred_baseline = models.baseline.predict(X_val)
    
    assert train_pred_baseline.shape == (100,)
    assert val_pred_baseline.shape == (50,)


def test_xgb_models_with_constant_target():
    """Test XGBoost models behavior with constant target."""
    try:
        from src.models.xgb_models import train_xgb_models
    except ImportError:
        pytest.skip("xgboost is not installed")
    
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(100, 3))
    y_train = np.ones(100) * 5.0  # Constant target
    
    X_val = rng.normal(size=(40, 3))
    y_val = np.ones(40) * 5.0
    
    models, metrics = train_xgb_models(X_train, y_train, X_val, y_val)
    
    # Should complete without errors
    assert "val_mae" in metrics["xgb_baseline"]
    # MAE should be very small (near zero)
    assert metrics["xgb_baseline"]["val_mae"] < 0.1
