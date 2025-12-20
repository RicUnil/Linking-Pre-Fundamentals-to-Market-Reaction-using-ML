"""
Unit tests for classification visualization functions (Step 20).

This module tests the classification plotting functions to ensure
they generate valid PNG files with dummy data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.visualization.classification_plots import (
    plot_roc_curves_multi_model,
    plot_confusion_matrix,
    plot_bar_auc_by_model,
)


def test_plot_roc_curves_multi_model(tmp_path: Path) -> None:
    """Test ROC curve plotting with multiple models."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    proba_by_model = {
        "model_a": np.array([0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.85]),
        "model_b": np.array([0.2, 0.7, 0.4, 0.85, 0.3, 0.6, 0.5, 0.8]),
    }
    out_path = tmp_path / "roc.png"
    
    plot_roc_curves_multi_model(y_true, proba_by_model, out_path, "Test ROC")
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_roc_curves_multi_model_with_none(tmp_path: Path) -> None:
    """Test ROC curve plotting when some models have None probabilities."""
    y_true = np.array([0, 1, 0, 1])
    proba_by_model = {
        "model_a": np.array([0.1, 0.8, 0.3, 0.9]),
        "model_b": None,  # Model without probability estimates
    }
    out_path = tmp_path / "roc_with_none.png"
    
    plot_roc_curves_multi_model(y_true, proba_by_model, out_path, "Test ROC with None")
    
    assert out_path.exists()


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    """Test confusion matrix plotting (raw counts)."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1])
    out_path = tmp_path / "cm.png"
    
    plot_confusion_matrix(y_true, y_pred, out_path, "Test CM", normalize=False)
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_confusion_matrix_normalized(tmp_path: Path) -> None:
    """Test normalized confusion matrix plotting."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1])
    out_path = tmp_path / "cm_normalized.png"
    
    plot_confusion_matrix(y_true, y_pred, out_path, "Test CM Normalized", normalize=True)
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_confusion_matrix_with_labels(tmp_path: Path) -> None:
    """Test confusion matrix with custom labels."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    out_path = tmp_path / "cm_labels.png"
    labels = ["Negative", "Positive"]
    
    plot_confusion_matrix(
        y_true, y_pred, out_path, "Test CM with Labels", normalize=False, labels=labels
    )
    
    assert out_path.exists()


def test_plot_bar_auc_by_model(tmp_path: Path) -> None:
    """Test AUC bar plot generation."""
    df = pd.DataFrame(
        {
            "model_name": ["model_a", "model_b", "model_c"],
            "split": ["val", "val", "val"],
            "roc_auc": [0.50, 0.55, 0.60],
        }
    )
    out_path = tmp_path / "auc_bar.png"
    
    plot_bar_auc_by_model(df, split="val", out_path=out_path, title="Test AUC bar")
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_bar_auc_by_model_empty_split(tmp_path: Path) -> None:
    """Test AUC bar plot with empty split (should not create file)."""
    df = pd.DataFrame(
        {
            "model_name": ["model_a", "model_b"],
            "split": ["val", "val"],
            "roc_auc": [0.5, 0.6],
        }
    )
    out_path = tmp_path / "auc_bar_empty.png"
    
    # Request a split that doesn't exist
    plot_bar_auc_by_model(df, split="test", out_path=out_path, title="Test Empty")
    
    # Should not create file for empty split
    assert not out_path.exists()


def test_step_20_output_files_exist():
    """Test that Step 20 creates all expected output files."""
    from src.config import Settings
    
    settings = Settings()
    output_dir = settings.RESULTS_DIR / "step_20" / "figures_classification"
    
    if not output_dir.exists():
        pytest.skip("Step 20 not run yet")
    
    # Check for expected files
    expected_files = [
        "roc_curves_val.png",
        "roc_curves_test.png",
        "auc_by_model_val.png",
        "auc_by_model_test.png",
    ]
    
    for filename in expected_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"Expected file {filename} not found"
        assert filepath.stat().st_size > 0, f"File {filename} is empty"


def test_step_20_confusion_matrices_exist():
    """Test that Step 20 creates confusion matrix files."""
    from src.config import Settings
    import json
    
    settings = Settings()
    output_dir = settings.RESULTS_DIR / "step_20" / "figures_classification"
    
    if not output_dir.exists():
        pytest.skip("Step 20 not run yet")
    
    # Find confusion matrix files (pattern: confusion_matrix_*_*.png)
    cm_files = list(output_dir.glob("confusion_matrix_*.png"))
    
    # Should have at least 4 confusion matrices (val, test, val_normalized, test_normalized)
    assert len(cm_files) >= 4, f"Expected at least 4 confusion matrix files, found {len(cm_files)}"
    
    # Check that files are not empty
    for cm_file in cm_files:
        assert cm_file.stat().st_size > 0, f"Confusion matrix file {cm_file.name} is empty"


def test_step_20_completion_marker_exists():
    """Test that Step 20 creates a completion marker."""
    from src.config import Settings
    
    settings = Settings()
    marker_path = settings.RESULTS_DIR / "step_20" / "step_20_completed.txt"
    
    if not marker_path.exists():
        pytest.skip("Step 20 not run yet")
    
    assert marker_path.exists()
    
    # Check that marker contains expected content
    content = marker_path.read_text()
    assert "Step 20 completed" in content
    assert "classification figures" in content
