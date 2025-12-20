"""
Unit tests for dataset preprocessing and splitting logic.

This module tests the functions used to prepare ML-ready datasets,
including time-based splitting and feature scaling.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.preprocessing.dataset import (
    DatasetSpec,
    build_dataset_spec,
    clean_and_filter_rows,
    train_val_test_split_time_based,
    scale_and_package_matrices,
)


def test_build_dataset_spec_identifies_features():
    """Test that build_dataset_spec correctly identifies feature columns."""
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "ticker"
    settings.EARNINGS_DATE_COLUMN = "earnings_date"
    
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "earnings_date": ["2019-01-01", "2019-02-01"],
        "f_eps_surprise": [0.1, 0.2],
        "f_revenue_growth": [0.05, 0.03],
        "stock_momentum_1m": [0.02, -0.01],
        "pre_volatility_30d": [0.15, 0.18],
        "excess_return_30d": [0.05, -0.02],
        "label_outperform_30d": [1, 0],
        "pre_window_start": ["2018-12-01", "2019-01-01"],
        "has_full_post_window": [True, True],
    })
    
    spec = build_dataset_spec(df, settings)
    
    # Should include fundamental and market features
    assert "f_eps_surprise" in spec.feature_columns
    assert "f_revenue_growth" in spec.feature_columns
    assert "stock_momentum_1m" in spec.feature_columns
    assert "pre_volatility_30d" in spec.feature_columns
    
    # Should exclude identifiers, targets, and metadata
    assert "ticker" not in spec.feature_columns
    assert "earnings_date" not in spec.feature_columns
    assert "excess_return_30d" not in spec.feature_columns
    assert "label_outperform_30d" not in spec.feature_columns
    assert "pre_window_start" not in spec.feature_columns
    assert "has_full_post_window" not in spec.feature_columns
    
    # Should have correct targets
    assert spec.target_regression == "excess_return_30d"
    assert spec.target_classification == "label_outperform_30d"


def test_clean_and_filter_rows_removes_missing_targets():
    """Test that clean_and_filter_rows removes rows with missing targets."""
    spec = DatasetSpec(
        id_columns=["ticker"],
        feature_columns=["f_dummy"],
        target_regression="excess_return_30d",
        target_classification="label_outperform_30d",
    )
    
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "f_dummy": [1.0, 2.0, 3.0, 4.0],
        "excess_return_30d": [0.1, np.nan, 0.3, 0.4],
        "label_outperform_30d": [1, 0, np.nan, 1],
        "has_full_post_window": [True, True, True, True],
    })
    
    df_clean = clean_and_filter_rows(df, spec)
    
    # Should keep only rows with both targets non-null
    assert len(df_clean) == 2
    assert df_clean["ticker"].tolist() == ["A", "D"]


def test_clean_and_filter_rows_requires_post_window():
    """Test that clean_and_filter_rows requires full post-window coverage."""
    spec = DatasetSpec(
        id_columns=["ticker"],
        feature_columns=["f_dummy"],
        target_regression="excess_return_30d",
        target_classification="label_outperform_30d",
    )
    
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "f_dummy": [1.0, 2.0, 3.0],
        "excess_return_30d": [0.1, 0.2, 0.3],
        "label_outperform_30d": [1, 0, 1],
        "has_full_post_window": [True, False, True],
    })
    
    df_clean = clean_and_filter_rows(df, spec)
    
    # Should keep only rows with has_full_post_window == True
    assert len(df_clean) == 2
    assert df_clean["ticker"].tolist() == ["A", "C"]


def test_time_based_split_respects_cutoff():
    """Test that time-based split correctly separates train/val/test by date."""
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "ticker"
    settings.EARNINGS_DATE_COLUMN = "earnings_date"
    
    dates = pd.to_datetime([
        "2018-01-01", "2018-06-01", "2019-06-01",
        "2019-12-31", "2020-01-01", "2021-01-01"
    ])
    
    df = pd.DataFrame({
        "ticker": ["A", "A", "A", "A", "A", "A"],
        "earnings_date": dates,
        "f_dummy": [1, 2, 3, 4, 5, 6],
        "excess_return_30d": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "label_outperform_30d": [1, 0, 1, 0, 1, 0],
    })
    
    spec = DatasetSpec(
        id_columns=["ticker", "earnings_date"],
        feature_columns=["f_dummy"],
        target_regression="excess_return_30d",
        target_classification="label_outperform_30d",
    )
    
    splits = train_val_test_split_time_based(
        df, spec, settings,
        test_split_date="2020-01-01",
        val_fraction_within_train=0.5
    )
    
    # Test set should have dates >= 2020-01-01
    assert all(splits["test"]["earnings_date"] >= pd.Timestamp("2020-01-01"))
    assert len(splits["test"]) == 2  # 2020-01-01 and 2021-01-01
    
    # Train and val should have dates < 2020-01-01
    assert all(splits["train"]["earnings_date"] < pd.Timestamp("2020-01-01"))
    assert all(splits["val"]["earnings_date"] < pd.Timestamp("2020-01-01"))
    
    # Val should come after train (time-based)
    if len(splits["train"]) > 0 and len(splits["val"]) > 0:
        train_max = splits["train"]["earnings_date"].max()
        val_min = splits["val"]["earnings_date"].min()
        assert train_max <= val_min


def test_time_based_split_preserves_temporal_order():
    """Test that validation set comes after training set in time."""
    settings = Settings()
    settings.EARNINGS_TICKER_COLUMN = "ticker"
    settings.EARNINGS_DATE_COLUMN = "earnings_date"
    
    dates = pd.to_datetime([
        "2017-01-01", "2017-06-01", "2018-01-01",
        "2018-06-01", "2019-01-01", "2019-06-01"
    ])
    
    df = pd.DataFrame({
        "ticker": ["A"] * 6,
        "earnings_date": dates,
        "f_dummy": [1, 2, 3, 4, 5, 6],
        "excess_return_30d": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "label_outperform_30d": [1, 0, 1, 0, 1, 0],
    })
    
    spec = DatasetSpec(
        id_columns=["ticker", "earnings_date"],
        feature_columns=["f_dummy"],
        target_regression="excess_return_30d",
        target_classification="label_outperform_30d",
    )
    
    splits = train_val_test_split_time_based(
        df, spec, settings,
        test_split_date="2020-01-01",
        val_fraction_within_train=0.33
    )
    
    # All should be in train+val (pre-2020)
    assert len(splits["test"]) == 0
    assert len(splits["train"]) + len(splits["val"]) == 6
    
    # Validation should come after training
    train_max = splits["train"]["earnings_date"].max()
    val_min = splits["val"]["earnings_date"].min()
    assert train_max <= val_min


def test_scaler_fits_on_train_only():
    """Test that StandardScaler is fitted on training data only."""
    spec = DatasetSpec(
        id_columns=["id"],
        feature_columns=["x"],
        target_regression="y",
        target_classification="y_cls",
    )
    
    train = pd.DataFrame({
        "id": [1, 2],
        "x": [0.0, 2.0],
        "y": [0.0, 1.0],
        "y_cls": [0, 1]
    })
    val = pd.DataFrame({
        "id": [3],
        "x": [1.0],
        "y": [0.5],
        "y_cls": [1]
    })
    test = pd.DataFrame({
        "id": [4],
        "x": [3.0],
        "y": [1.5],
        "y_cls": [1]
    })
    
    splits = {"train": train, "val": val, "test": test}
    
    X_splits, y_splits, scaler = scale_and_package_matrices(splits, spec)
    
    # After scaling: train x should have mean ~ 0, std ~ 1
    train_scaled = X_splits["train"].ravel()
    assert np.isclose(train_scaled.mean(), 0.0, atol=1e-6)
    assert np.isclose(train_scaled.std(ddof=0), 1.0, atol=1e-6)
    
    # Scaler should have been fitted on train data (mean=1.0, std=1.0)
    assert np.isclose(scaler.mean_[0], 1.0, atol=1e-6)
    assert np.isclose(scaler.scale_[0], 1.0, atol=1e-6)


def test_scaler_transforms_val_and_test():
    """Test that scaler correctly transforms validation and test sets."""
    spec = DatasetSpec(
        id_columns=["id"],
        feature_columns=["x"],
        target_regression="y",
        target_classification="y_cls",
    )
    
    # Train: x = [0, 10], mean=5, std=5
    train = pd.DataFrame({
        "id": [1, 2],
        "x": [0.0, 10.0],
        "y": [0.0, 1.0],
        "y_cls": [0, 1]
    })
    
    # Val: x = [5] should become (5-5)/5 = 0
    val = pd.DataFrame({
        "id": [3],
        "x": [5.0],
        "y": [0.5],
        "y_cls": [1]
    })
    
    # Test: x = [15] should become (15-5)/5 = 2
    test = pd.DataFrame({
        "id": [4],
        "x": [15.0],
        "y": [1.5],
        "y_cls": [1]
    })
    
    splits = {"train": train, "val": val, "test": test}
    
    X_splits, y_splits, scaler = scale_and_package_matrices(splits, spec)
    
    # Check transformations
    assert np.isclose(X_splits["val"][0, 0], 0.0, atol=1e-6)
    assert np.isclose(X_splits["test"][0, 0], 2.0, atol=1e-6)


def test_scale_and_package_preserves_targets():
    """Test that target arrays are correctly extracted and preserved."""
    spec = DatasetSpec(
        id_columns=["id"],
        feature_columns=["x1", "x2"],
        target_regression="y",
        target_classification="y_cls",
    )
    
    train = pd.DataFrame({
        "id": [1, 2],
        "x1": [1.0, 2.0],
        "x2": [3.0, 4.0],
        "y": [0.1, 0.2],
        "y_cls": [0, 1]
    })
    
    val = pd.DataFrame({
        "id": [3],
        "x1": [1.5],
        "x2": [3.5],
        "y": [0.15],
        "y_cls": [1]
    })
    
    test = pd.DataFrame({
        "id": [4],
        "x1": [2.5],
        "x2": [4.5],
        "y": [0.25],
        "y_cls": [1]
    })
    
    splits = {"train": train, "val": val, "test": test}
    
    X_splits, y_splits, scaler = scale_and_package_matrices(splits, spec)
    
    # Check target arrays
    assert np.allclose(y_splits["train"], [0.1, 0.2])
    assert np.allclose(y_splits["val"], [0.15])
    assert np.allclose(y_splits["test"], [0.25])
    
    # Check feature matrix shapes
    assert X_splits["train"].shape == (2, 2)
    assert X_splits["val"].shape == (1, 2)
    assert X_splits["test"].shape == (1, 2)
