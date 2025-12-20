"""
Regression evaluation metrics.

This module provides functions to compute standard regression metrics
for model evaluation and comparison.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute standard regression metrics: MAE, RMSE, R².

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_pred : np.ndarray
        Predicted target values.
    prefix : str, optional
        Optional prefix for metric names (e.g., 'train_', 'val_').

    Returns
    -------
    dict
        Dictionary with keys like '{prefix}mae', '{prefix}rmse', '{prefix}r2'.
        
    Examples
    --------
    >>> y_true = np.array([0.0, 1.0, 2.0])
    >>> y_pred = np.array([0.1, 0.9, 2.1])
    >>> metrics = regression_metrics(y_true, y_pred, prefix='val_')
    >>> 'val_mae' in metrics
    True
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    base = prefix or ""
    return {
        f"{base}mae": float(mae),
        f"{base}rmse": float(rmse),
        f"{base}r2": float(r2),
    }
