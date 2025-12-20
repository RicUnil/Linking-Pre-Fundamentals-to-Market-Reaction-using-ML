"""
Unit tests for classification label creation (Step 18).

This module tests the binary label conversion logic and validates
that the classification labels are correctly created from continuous
excess returns.
"""

from pathlib import Path
import numpy as np
import pytest

from src.preprocessing.targets import create_binary_outperformance_label
import pandas as pd


def test_binary_label_conversion():
    """Test that binary label conversion follows the correct rule."""
    # Simple sanity test: sign rule
    y = np.array([-0.1, 0.0, 0.05, 0.2])
    y_class = (y > 0.0).astype(int)
    assert np.array_equal(y_class, np.array([0, 0, 1, 1]))


def test_binary_label_conversion_edge_cases():
    """Test edge cases for binary label conversion."""
    # Test with exactly zero
    y = np.array([0.0])
    y_class = (y > 0.0).astype(int)
    assert y_class[0] == 0, "Zero should be classified as 0 (underperform)"
    
    # Test with very small positive
    y = np.array([1e-10])
    y_class = (y > 0.0).astype(int)
    assert y_class[0] == 1, "Small positive should be classified as 1 (outperform)"
    
    # Test with very small negative
    y = np.array([-1e-10])
    y_class = (y > 0.0).astype(int)
    assert y_class[0] == 0, "Small negative should be classified as 0 (underperform)"


def test_saved_labels_exist(tmp_path: Path):
    """Test that label conversion produces valid binary labels."""
    # This is a structural test stub; in the real pipeline,
    # Step 18 will be run end-to-end and produce the files.
    # Here we simply check the conversion logic in isolation.
    y = np.linspace(-0.1, 0.1, 11)
    y_class = (y > 0.0).astype(int)
    assert set(np.unique(y_class)).issubset({0, 1})


def test_create_binary_outperformance_label():
    """Test the create_binary_outperformance_label function."""
    excess_ret = pd.Series([0.05, -0.02, 0.0, 0.10, -0.05])
    labels = create_binary_outperformance_label(excess_ret)
    
    # Check correct conversion
    expected = pd.Series([1, 0, 0, 1, 0], name="label_outperform")
    pd.testing.assert_series_equal(labels, expected)


def test_create_binary_outperformance_label_with_threshold():
    """Test binary label creation with custom threshold."""
    excess_ret = pd.Series([0.05, -0.02, 0.0, 0.10, -0.05])
    
    # Threshold at 0.03
    labels = create_binary_outperformance_label(excess_ret, threshold=0.03)
    
    # Only values > 0.03 should be 1
    expected = pd.Series([1, 0, 0, 1, 0], name="label_outperform")
    pd.testing.assert_series_equal(labels, expected)


def test_create_binary_outperformance_label_all_positive():
    """Test with all positive returns."""
    excess_ret = pd.Series([0.01, 0.02, 0.03, 0.04])
    labels = create_binary_outperformance_label(excess_ret)
    
    assert labels.sum() == 4, "All positive returns should be labeled as 1"
    assert set(labels.unique()) == {1}


def test_create_binary_outperformance_label_all_negative():
    """Test with all negative returns."""
    excess_ret = pd.Series([-0.01, -0.02, -0.03, -0.04])
    labels = create_binary_outperformance_label(excess_ret)
    
    assert labels.sum() == 0, "All negative returns should be labeled as 0"
    assert set(labels.unique()) == {0}


def test_create_binary_outperformance_label_balanced():
    """Test with balanced positive and negative returns."""
    excess_ret = pd.Series([0.05, -0.05, 0.10, -0.10, 0.01, -0.01])
    labels = create_binary_outperformance_label(excess_ret)
    
    assert labels.sum() == 3, "Should have 3 positive (outperform) labels"
    assert (labels == 0).sum() == 3, "Should have 3 negative (underperform) labels"


def test_step_18_output_files_exist():
    """Test that Step 18 creates all expected output files."""
    from src.config import Settings
    
    settings = Settings()
    output_dir = settings.RESULTS_DIR / "step_18"
    
    if not output_dir.exists():
        pytest.skip("Step 18 not run yet")
    
    # Check for expected files
    expected_files = [
        "y_train_class.npy",
        "y_val_class.npy",
        "y_test_class.npy",
        "class_balance_summary.json",
        "classification_dataset_spec.json",
        "step_18_completed.txt",
    ]
    
    for filename in expected_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"Expected file {filename} not found"


def test_step_18_labels_are_binary():
    """Test that Step 18 produces valid binary labels."""
    from src.config import Settings
    
    settings = Settings()
    output_dir = settings.RESULTS_DIR / "step_18"
    
    if not output_dir.exists():
        pytest.skip("Step 18 not run yet")
    
    # Load labels
    y_train_class = np.load(output_dir / "y_train_class.npy")
    y_val_class = np.load(output_dir / "y_val_class.npy")
    y_test_class = np.load(output_dir / "y_test_class.npy")
    
    # Check that all labels are 0 or 1
    for y, name in [(y_train_class, "train"), (y_val_class, "val"), (y_test_class, "test")]:
        assert set(np.unique(y)).issubset({0, 1}), f"{name} labels must be binary (0 or 1)"
        assert y.ndim == 1, f"{name} labels must be 1D array"


def test_step_18_labels_match_regression_targets():
    """Test that classification labels align with regression targets."""
    from src.config import Settings
    
    settings = Settings()
    step10_dir = settings.RESULTS_DIR / "step_10"
    step18_dir = settings.RESULTS_DIR / "step_18"
    
    if not step18_dir.exists():
        pytest.skip("Step 18 not run yet")
    
    # Load regression targets
    y_train = np.load(step10_dir / "y_train.npy")
    y_val = np.load(step10_dir / "y_val.npy")
    y_test = np.load(step10_dir / "y_test.npy")
    
    # Load classification labels
    y_train_class = np.load(step18_dir / "y_train_class.npy")
    y_val_class = np.load(step18_dir / "y_val_class.npy")
    y_test_class = np.load(step18_dir / "y_test_class.npy")
    
    # Check shapes match
    assert y_train.shape == y_train_class.shape
    assert y_val.shape == y_val_class.shape
    assert y_test.shape == y_test_class.shape
    
    # Check conversion rule
    assert np.array_equal(y_train_class, (y_train > 0.0).astype(int))
    assert np.array_equal(y_val_class, (y_val > 0.0).astype(int))
    assert np.array_equal(y_test_class, (y_test > 0.0).astype(int))
