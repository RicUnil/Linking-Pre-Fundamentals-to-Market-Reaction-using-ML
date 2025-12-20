"""
Unit tests for baseline and linear regression models.

This module tests the baseline models (mean and CAPM) and linear
regression models to ensure correct behavior.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.models.baselines import (
    train_mean_baseline,
    estimate_capm_betas,
    predict_capm_baseline,
)
from src.models.linear_models import train_linear_models
from src.metrics.regression import regression_metrics


def test_mean_baseline_predicts_mean():
    """Test that mean baseline predicts the training set mean."""
    y_train = np.array([0.0, 1.0, 2.0])
    y_val = np.array([10.0, 20.0])
    
    model, y_train_pred, y_val_pred = train_mean_baseline(y_train, y_val)
    
    # All predictions should equal the training mean
    expected_mean = y_train.mean()
    assert np.allclose(y_train_pred, expected_mean)
    assert np.allclose(y_val_pred, expected_mean)
    
    # Check shapes
    assert y_train_pred.shape == y_train.shape
    assert y_val_pred.shape == y_val.shape


def test_mean_baseline_with_negative_values():
    """Test mean baseline with negative values."""
    y_train = np.array([-1.0, 0.0, 1.0])
    y_val = np.array([5.0])
    
    model, y_train_pred, y_val_pred = train_mean_baseline(y_train, y_val)
    
    expected_mean = 0.0
    assert np.allclose(y_train_pred, expected_mean)
    assert np.allclose(y_val_pred, expected_mean)


def test_capm_beta_estimation_basic():
    """Test basic CAPM beta estimation."""
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "ticker"
    
    # Create synthetic data with positive correlation
    df_train = pd.DataFrame({
        "ticker": ["A"] * 20,
        "excess_return_30d": np.linspace(-0.1, 0.1, 20),
        "spy_pre_return_30d": np.linspace(-0.05, 0.05, 20),
    })
    
    betas = estimate_capm_betas(df_train, settings)
    
    # Should have one ticker
    assert "A" in betas.index
    assert "beta" in betas.columns
    assert "alpha" in betas.columns
    
    # Beta should be positive (positive correlation)
    assert betas.loc["A", "beta"] > 0


def test_capm_beta_estimation_multiple_tickers():
    """Test CAPM beta estimation with multiple tickers."""
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "ticker"
    
    # Create data for two tickers
    df_train = pd.DataFrame({
        "ticker": ["A"] * 15 + ["B"] * 15,
        "excess_return_30d": np.concatenate([
            np.linspace(-0.1, 0.1, 15),
            np.linspace(0.1, -0.1, 15),
        ]),
        "spy_pre_return_30d": np.concatenate([
            np.linspace(-0.05, 0.05, 15),
            np.linspace(-0.05, 0.05, 15),
        ]),
    })
    
    betas = estimate_capm_betas(df_train, settings)
    
    # Should have both tickers
    assert len(betas) == 2
    assert "A" in betas.index
    assert "B" in betas.index


def test_capm_beta_estimation_insufficient_data():
    """Test that tickers with insufficient data are excluded."""
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "ticker"
    
    # Create data with only 5 observations (< 10 required)
    df_train = pd.DataFrame({
        "ticker": ["A"] * 5,
        "excess_return_30d": [0.01, 0.02, 0.03, 0.04, 0.05],
        "spy_pre_return_30d": [0.005, 0.010, 0.015, 0.020, 0.025],
    })
    
    betas = estimate_capm_betas(df_train, settings)
    
    # Should return empty dataframe (insufficient data)
    assert len(betas) == 0


def test_predict_capm_baseline():
    """Test CAPM baseline prediction."""
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "ticker"
    
    # Create beta dataframe
    betas = pd.DataFrame({
        "beta": [1.5],
        "alpha": [0.01],
    }, index=["A"])
    
    # Create test dataframe
    df = pd.DataFrame({
        "ticker": ["A", "A"],
        "spy_pre_return_30d": [0.02, -0.01],
        "excess_return_30d": [0.0, 0.0],  # Not used for prediction
    })
    
    preds = predict_capm_baseline(df, betas, settings)
    
    # Check predictions: alpha + beta * spy_return
    expected_0 = 0.01 + 1.5 * 0.02  # = 0.04
    expected_1 = 0.01 + 1.5 * (-0.01)  # = -0.005
    
    assert np.isclose(preds[0], expected_0, atol=1e-6)
    assert np.isclose(preds[1], expected_1, atol=1e-6)


def test_predict_capm_baseline_missing_ticker():
    """Test CAPM prediction with missing ticker (should use global mean)."""
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "ticker"
    
    # Betas only for ticker A
    betas = pd.DataFrame({
        "beta": [1.0],
        "alpha": [0.0],
    }, index=["A"])
    
    # Test with ticker B (not in betas)
    df = pd.DataFrame({
        "ticker": ["B"],
        "spy_pre_return_30d": [0.02],
        "excess_return_30d": [0.05],  # Global mean
    })
    
    preds = predict_capm_baseline(df, betas, settings)
    
    # Should fall back to global mean
    assert np.isclose(preds[0], 0.05, atol=1e-6)


def test_linear_models_train_and_return_metrics():
    """Test that linear models train and return proper metrics."""
    # Create synthetic data with linear relationship
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = X_train @ np.array([0.5, -0.2, 0.1]) + 0.01 * np.random.randn(100)
    
    X_val = np.random.randn(40, 3)
    y_val = X_val @ np.array([0.5, -0.2, 0.1]) + 0.01 * np.random.randn(40)
    
    models, metrics = train_linear_models(X_train, y_train, X_val, y_val)
    
    # Check that all models are present
    assert "linear" in metrics
    assert "ridge" in metrics
    assert "lasso" in metrics
    
    # Check that all metrics are present
    for model_name in ["linear", "ridge", "lasso"]:
        assert "train_mae" in metrics[model_name]
        assert "train_rmse" in metrics[model_name]
        assert "train_r2" in metrics[model_name]
        assert "val_mae" in metrics[model_name]
        assert "val_rmse" in metrics[model_name]
        assert "val_r2" in metrics[model_name]
    
    # Check that models are fitted
    assert hasattr(models.linear, 'coef_')
    assert hasattr(models.ridge, 'coef_')
    assert hasattr(models.lasso, 'coef_')


def test_linear_models_reasonable_performance():
    """Test that linear models achieve reasonable performance on synthetic data."""
    # Create data with strong linear relationship
    np.random.seed(42)
    X_train = np.random.randn(200, 5)
    true_coef = np.array([1.0, -0.5, 0.3, 0.0, 0.2])
    y_train = X_train @ true_coef + 0.05 * np.random.randn(200)
    
    X_val = np.random.randn(50, 5)
    y_val = X_val @ true_coef + 0.05 * np.random.randn(50)
    
    models, metrics = train_linear_models(X_train, y_train, X_val, y_val)
    
    # Linear model should achieve high R² on this synthetic data
    assert metrics["linear"]["val_r2"] > 0.8
    
    # MAE should be small
    assert metrics["linear"]["val_mae"] < 0.2


def test_regression_metrics_computation():
    """Test regression metrics computation."""
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.1, 0.9, 2.1, 2.9])
    
    metrics = regression_metrics(y_true, y_pred, prefix="test_")
    
    # Check that all metrics are present
    assert "test_mae" in metrics
    assert "test_rmse" in metrics
    assert "test_r2" in metrics
    
    # Check that values are reasonable
    assert metrics["test_mae"] > 0
    assert metrics["test_rmse"] > 0
    assert 0 <= metrics["test_r2"] <= 1


def test_regression_metrics_perfect_prediction():
    """Test regression metrics with perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    
    metrics = regression_metrics(y_true, y_pred)
    
    # Perfect prediction should have MAE=0, RMSE=0, R²=1
    assert np.isclose(metrics["mae"], 0.0, atol=1e-10)
    assert np.isclose(metrics["rmse"], 0.0, atol=1e-10)
    assert np.isclose(metrics["r2"], 1.0, atol=1e-10)
