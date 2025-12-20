"""
Unit tests for tree-based regression models.

This module tests the tree-based models (Random Forest, Gradient Boosting,
HistGradientBoosting) to ensure correct behavior.
"""

import numpy as np
import pytest

from src.models.tree_models import train_tree_models


def test_tree_models_train_and_return_metrics():
    """Test that tree models train and return proper metrics."""
    # Simple synthetic dataset with linear relationship
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(200, 5))
    true_coef = np.array([0.5, -0.3, 0.2, 0.0, 0.1])
    y_train = X_train @ true_coef + 0.01 * rng.normal(size=200)
    
    X_val = rng.normal(size=(80, 5))
    y_val = X_val @ true_coef + 0.01 * rng.normal(size=80)
    
    models, metrics = train_tree_models(X_train, y_train, X_val, y_val)
    
    # Check that all tree models are present
    assert "tree_rf" in metrics
    assert "tree_gbr" in metrics
    assert "tree_hgb" in metrics
    
    # Check that all metrics are present for each model
    for model_name in ["tree_rf", "tree_gbr", "tree_hgb"]:
        assert "train_mae" in metrics[model_name]
        assert "train_rmse" in metrics[model_name]
        assert "train_r2" in metrics[model_name]
        assert "val_mae" in metrics[model_name]
        assert "val_rmse" in metrics[model_name]
        assert "val_r2" in metrics[model_name]
    
    # Check that models are fitted
    assert hasattr(models.random_forest, 'feature_importances_')
    assert hasattr(models.gradient_boosting, 'feature_importances_')
    # HistGradientBoosting may not have feature_importances_ in older sklearn
    # Just check it's fitted by checking it can predict
    assert models.hist_gradient_boosting.predict(X_val).shape == y_val.shape


def test_tree_models_reasonable_performance():
    """Test that tree models achieve reasonable performance on synthetic data."""
    # Create data with strong relationship
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(300, 5))
    true_coef = np.array([1.0, -0.5, 0.3, 0.0, 0.2])
    y_train = X_train @ true_coef + 0.05 * rng.normal(size=300)
    
    X_val = rng.normal(size=(100, 5))
    y_val = X_val @ true_coef + 0.05 * rng.normal(size=100)
    
    models, metrics = train_tree_models(X_train, y_train, X_val, y_val)
    
    # Tree models should achieve reasonable R² on this synthetic data
    # (may not be as high as linear models for linear data, but should be positive)
    assert metrics["tree_rf"]["val_r2"] > 0.5
    assert metrics["tree_gbr"]["val_r2"] > 0.5
    assert metrics["tree_hgb"]["val_r2"] > 0.5


def test_tree_models_handle_more_samples_than_features():
    """Test that tree models work with more samples than features."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(500, 3))
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + 0.1 * rng.normal(size=500)
    
    X_val = rng.normal(size=(100, 3))
    y_val = X_val[:, 0] + 0.5 * X_val[:, 1] + 0.1 * rng.normal(size=100)
    
    models, metrics = train_tree_models(X_train, y_train, X_val, y_val)
    
    # Should complete without errors
    assert len(metrics) == 3
    assert all("val_mae" in m for m in metrics.values())


def test_tree_models_feature_importances():
    """Test that tree models compute feature importances."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(200, 5))
    # Make first feature very predictive
    y_train = 2.0 * X_train[:, 0] + 0.01 * rng.normal(size=200)
    
    X_val = rng.normal(size=(80, 5))
    y_val = 2.0 * X_val[:, 0] + 0.01 * rng.normal(size=80)
    
    models, metrics = train_tree_models(X_train, y_train, X_val, y_val)
    
    # Check that feature importances exist and sum to ~1
    # Only check for RF and GBR (HistGradientBoosting may not have this in older sklearn)
    for model in [models.random_forest, models.gradient_boosting]:
        importances = model.feature_importances_
        assert len(importances) == 5
        assert np.isclose(importances.sum(), 1.0, atol=0.01)
        # First feature should have highest importance
        assert importances[0] > 0.5


def test_tree_models_predictions_shape():
    """Test that predictions have correct shape."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(100, 4))
    y_train = rng.normal(size=100)
    
    X_val = rng.normal(size=(50, 4))
    y_val = rng.normal(size=50)
    
    models, metrics = train_tree_models(X_train, y_train, X_val, y_val)
    
    # Test predictions
    train_pred_rf = models.random_forest.predict(X_train)
    val_pred_rf = models.random_forest.predict(X_val)
    
    assert train_pred_rf.shape == (100,)
    assert val_pred_rf.shape == (50,)


def test_tree_models_with_constant_target():
    """Test tree models behavior with constant target."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(100, 3))
    y_train = np.ones(100) * 5.0  # Constant target
    
    X_val = rng.normal(size=(40, 3))
    y_val = np.ones(40) * 5.0
    
    models, metrics = train_tree_models(X_train, y_train, X_val, y_val)
    
    # Should complete without errors
    # R² should be undefined or 1.0 for constant target
    assert "val_mae" in metrics["tree_rf"]
    # MAE should be very small (near zero)
    assert metrics["tree_rf"]["val_mae"] < 0.1
