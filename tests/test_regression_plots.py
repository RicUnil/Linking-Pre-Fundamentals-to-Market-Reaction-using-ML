"""
Unit tests for regression visualization functions (Step 17).

This module tests the plotting utilities without requiring full project data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.visualization.regression_plots import (
    plot_actual_vs_predicted_scatter,
    plot_residuals_histogram,
    plot_residuals_vs_predictions,
    plot_bar_metrics_by_model,
    plot_rolling_metric_over_time,
)


def test_basic_scatter_plot(tmp_path: Path) -> None:
    """Test that actual vs predicted scatter plot can be created."""
    y_true = np.array([0.0, 0.1, -0.1, 0.05, -0.05])
    y_pred = np.array([0.0, 0.08, -0.08, 0.06, -0.04])
    out_path = tmp_path / "scatter.png"
    
    plot_actual_vs_predicted_scatter(y_true, y_pred, out_path, "Test scatter")
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_basic_histogram_plot(tmp_path: Path) -> None:
    """Test that residual histogram can be created."""
    residuals = np.array([0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05])
    out_path = tmp_path / "hist.png"
    
    plot_residuals_histogram(residuals, out_path, "Test hist")
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_residuals_vs_predictions_plot(tmp_path: Path) -> None:
    """Test that residuals vs predictions scatter plot can be created."""
    y_pred = np.array([0.0, 0.1, -0.1, 0.05, -0.05])
    residuals = np.array([0.01, -0.02, 0.01, -0.01, 0.02])
    out_path = tmp_path / "res_vs_pred.png"
    
    plot_residuals_vs_predictions(y_pred, residuals, out_path, "Test residuals")
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_basic_bar_plot(tmp_path: Path) -> None:
    """Test that bar chart of metrics can be created."""
    df = pd.DataFrame(
        {
            "model_name": ["model_a", "model_b", "model_c"],
            "val_mae": [0.1, 0.2, 0.15],
        }
    )
    out_path = tmp_path / "bar.png"
    
    plot_bar_metrics_by_model(df, "val_mae", out_path, "Test bar", "Val MAE")
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_rolling_metric_plot(tmp_path: Path) -> None:
    """Test that rolling metric time-series plot can be created."""
    df = pd.DataFrame(
        {
            "fold_id": [1, 1, 2, 2, 3, 3],
            "test_start": ["2020-01-01", "2020-01-01", "2021-01-01", "2021-01-01", "2022-01-01", "2022-01-01"],
            "model_name": ["model_a", "model_b", "model_a", "model_b", "model_a", "model_b"],
            "test_mae": [0.05, 0.06, 0.055, 0.065, 0.052, 0.062],
        }
    )
    out_path = tmp_path / "rolling.png"
    
    plot_rolling_metric_over_time(
        df, "test_mae", out_path, "Test rolling", "Test MAE"
    )
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_rolling_metric_plot_with_model_subset(tmp_path: Path) -> None:
    """Test rolling metric plot with specific models."""
    df = pd.DataFrame(
        {
            "fold_id": [1, 1, 1, 2, 2, 2],
            "test_start": ["2020-01-01", "2020-01-01", "2020-01-01", "2021-01-01", "2021-01-01", "2021-01-01"],
            "model_name": ["model_a", "model_b", "model_c", "model_a", "model_b", "model_c"],
            "test_r2": [0.01, 0.02, 0.015, 0.012, 0.022, 0.017],
        }
    )
    out_path = tmp_path / "rolling_subset.png"
    
    plot_rolling_metric_over_time(
        df, "test_r2", out_path, "Test rolling subset", "Test R²",
        models_to_plot=["model_a", "model_c"]
    )
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_step_17_output_files_exist():
    """Test that Step 17 creates all expected output files."""
    from src.config import Settings
    
    settings = Settings()
    output_dir = settings.RESULTS_DIR / "step_17" / "figures"
    
    if not output_dir.exists():
        pytest.skip("Step 17 not run yet")
    
    # Check for expected files
    expected_files = [
        "actual_vs_pred_ridge_val.png",
        "actual_vs_pred_ridge_test.png",
        "actual_vs_pred_rf_test.png",
        "residuals_hist_ridge_test.png",
        "residuals_vs_pred_ridge_test.png",
        "val_mae_by_model.png",
        "val_r2_by_model.png",
        "test_mae_by_model.png",
        "test_r2_by_model.png",
        "rolling_test_mae_by_model.png",
        "rolling_test_r2_by_model.png",
    ]
    
    for filename in expected_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"Expected file {filename} not found"
        assert filepath.stat().st_size > 0, f"File {filename} is empty"


def test_step_17_completion_marker():
    """Test that Step 17 completion marker exists."""
    from src.config import Settings
    
    settings = Settings()
    marker_path = settings.RESULTS_DIR / "step_17" / "step_17_completed.txt"
    
    if not marker_path.exists():
        pytest.skip("Step 17 not run yet")
    
    assert marker_path.exists()
    
    # Check that marker contains expected content
    content = marker_path.read_text()
    assert "Step 17 completed" in content
    assert "regression figures generated" in content
