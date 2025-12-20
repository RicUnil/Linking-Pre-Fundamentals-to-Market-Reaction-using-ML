"""
Linear regression models for excess return prediction.

This module implements standard linear regression models:
- Ordinary Least Squares (LinearRegression)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import logging
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


logger = logging.getLogger(__name__)


@dataclass
class LinearModels:
    """
    Container for fitted linear regression models.
    
    Attributes
    ----------
    linear : LinearRegression
        Ordinary least squares regression.
    ridge : Ridge
        Ridge regression with L2 regularization.
    lasso : Lasso
        Lasso regression with L1 regularization.
    """
    
    linear: LinearRegression
    ridge: Ridge
    lasso: Lasso


def train_linear_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[LinearModels, Dict[str, Dict[str, float]]]:
    """
    Train linear, ridge and lasso regression models on the training data
    and evaluate on both train and validation sets.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (scaled).
    y_train : np.ndarray
        Training targets.
    X_val : np.ndarray
        Validation feature matrix (scaled).
    y_val : np.ndarray
        Validation targets.

    Returns
    -------
    models : LinearModels
        Container with fitted sklearn models.
    metrics : dict
        Nested dict metrics[model_name][split_name_metric] = value.
    """
    logger.info("=" * 70)
    logger.info("TRAINING LINEAR MODELS")
    logger.info("=" * 70)
    
    # Initialize models with reasonable hyperparameters
    models = LinearModels(
        linear=LinearRegression(),
        ridge=Ridge(alpha=1.0),
        lasso=Lasso(alpha=0.001, max_iter=10_000),
    )
    
    metrics: Dict[str, Dict[str, float]] = {}
    
    # Train and evaluate each model
    for name, model in [
        ("linear", models.linear),
        ("ridge", models.ridge),
        ("lasso", models.lasso),
    ]:
        logger.info(f"\nTraining {name.upper()} model...")
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Generate predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Compute metrics
        metrics[name] = {
            "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "train_r2": float(r2_score(y_train, y_train_pred)),
            "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
            "val_r2": float(r2_score(y_val, y_val_pred)),
        }
        
        logger.info(f"  Train MAE: {metrics[name]['train_mae']:.6f}")
        logger.info(f"  Train RMSE: {metrics[name]['train_rmse']:.6f}")
        logger.info(f"  Train R²: {metrics[name]['train_r2']:.6f}")
        logger.info(f"  Val MAE: {metrics[name]['val_mae']:.6f}")
        logger.info(f"  Val RMSE: {metrics[name]['val_rmse']:.6f}")
        logger.info(f"  Val R²: {metrics[name]['val_r2']:.6f}")
        
        # Log coefficient info for interpretability
        if hasattr(model, 'coef_'):
            n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
            logger.info(f"  Non-zero coefficients: {n_nonzero} / {len(model.coef_)}")
    
    return models, metrics
