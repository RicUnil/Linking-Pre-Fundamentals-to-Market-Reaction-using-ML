"""
Unit tests for classification model training (Step 19).

This module tests the classification model training functions and
validates that metrics are computed correctly.
"""

import numpy as np
import pytest

from src.models.classification_models import (
    compute_classification_metrics,
    train_classification_models,
    ClassificationMetrics,
)


def test_compute_classification_metrics_basic() -> None:
    """Test basic classification metrics computation."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.8, 0.4, 0.2])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    
    assert isinstance(metrics, ClassificationMetrics)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.balanced_accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_compute_classification_metrics_dict() -> None:
    """Test metrics conversion to dictionary."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.8, 0.4, 0.2])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_proba).to_dict()
    
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics


def test_compute_classification_metrics_perfect() -> None:
    """Test metrics with perfect predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    
    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0


def test_compute_classification_metrics_no_proba() -> None:
    """Test metrics when probabilities are not available."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_proba=None)
    
    assert 0.0 <= metrics.accuracy <= 1.0
    assert np.isnan(metrics.roc_auc)


def test_train_classification_models_shapes() -> None:
    """Test that training returns correct structure."""
    # Tiny synthetic dataset
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, size=100)
    X_val = np.random.randn(30, 5)
    y_val = np.random.randint(0, 2, size=30)
    X_test = np.random.randn(30, 5)
    y_test = np.random.randint(0, 2, size=30)

    models, metrics = train_classification_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Check models
    assert len(models) == 4
    assert "logistic_regression" in models
    assert "dummy_most_frequent" in models
    assert "random_forest_classifier" in models
    assert "gradient_boosting_classifier" in models
    
    # Check metrics structure
    assert "logistic_regression" in metrics
    assert "train" in metrics["logistic_regression"]
    assert "val" in metrics["logistic_regression"]
    assert "test" in metrics["logistic_regression"]
    assert "roc_auc" in metrics["logistic_regression"]["val"]


def test_train_classification_models_all_metrics() -> None:
    """Test that all expected metrics are computed."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, size=100)
    X_val = np.random.randn(30, 5)
    y_val = np.random.randint(0, 2, size=30)
    X_test = np.random.randn(30, 5)
    y_test = np.random.randint(0, 2, size=30)

    models, metrics = train_classification_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    expected_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    
    for model_name in models.keys():
        for split in ["train", "val", "test"]:
            for metric_name in expected_metrics:
                assert metric_name in metrics[model_name][split], \
                    f"Missing {metric_name} for {model_name} on {split}"


def test_train_classification_models_dummy_baseline() -> None:
    """Test that dummy classifier is included and produces valid metrics."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, size=100)
    X_val = np.random.randn(30, 5)
    y_val = np.random.randint(0, 2, size=30)
    X_test = np.random.randn(30, 5)
    y_test = np.random.randint(0, 2, size=30)

    models, metrics = train_classification_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    assert "dummy_most_frequent" in models
    assert "dummy_most_frequent" in metrics
    
    # Dummy should have valid metrics
    dummy_test_metrics = metrics["dummy_most_frequent"]["test"]
    assert 0.0 <= dummy_test_metrics["accuracy"] <= 1.0


def test_step_19_output_files_exist():
    """Test that Step 19 creates all expected output files."""
    from src.config import Settings
    
    settings = Settings()
    output_dir = settings.RESULTS_DIR / "step_19"
    
    if not output_dir.exists():
        pytest.skip("Step 19 not run yet")
    
    # Check for expected files
    expected_files = [
        "dummy_most_frequent.joblib",
        "logistic_regression.joblib",
        "random_forest_classifier.joblib",
        "gradient_boosting_classifier.joblib",
        "classification_metrics_summary.json",
        "classification_metrics_summary.csv",
        "step_19_completed.txt",
    ]
    
    for filename in expected_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"Expected file {filename} not found"


def test_step_19_metrics_are_valid():
    """Test that Step 19 produces valid metrics."""
    from src.config import Settings
    import json
    
    settings = Settings()
    output_dir = settings.RESULTS_DIR / "step_19"
    
    if not output_dir.exists():
        pytest.skip("Step 19 not run yet")
    
    # Load metrics
    metrics_path = output_dir / "classification_metrics_summary.json"
    with metrics_path.open("r") as f:
        metrics = json.load(f)
    
    # Check structure
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    
    # Check each model has valid metrics
    for model_name, splits in metrics.items():
        for split_name, split_metrics in splits.items():
            # Check all metrics are present
            assert "accuracy" in split_metrics
            assert "roc_auc" in split_metrics
            
            # Check metrics are in valid range
            assert 0.0 <= split_metrics["accuracy"] <= 1.0
            if not np.isnan(split_metrics["roc_auc"]):
                assert 0.0 <= split_metrics["roc_auc"] <= 1.0
