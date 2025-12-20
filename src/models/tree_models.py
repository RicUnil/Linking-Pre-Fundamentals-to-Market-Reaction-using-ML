"""
Tree-based regression models for excess return prediction.

This module implements ensemble tree-based models:
- Random Forest Regressor
- Gradient Boosting Regressor
- Histogram-based Gradient Boosting Regressor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import logging
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

from src.metrics.regression import regression_metrics


logger = logging.getLogger(__name__)


@dataclass
class TreeModels:
    """
    Container for fitted tree-based regression models.
    
    Attributes
    ----------
    random_forest : RandomForestRegressor
        Random forest ensemble model.
    gradient_boosting : GradientBoostingRegressor
        Gradient boosting model.
    hist_gradient_boosting : HistGradientBoostingRegressor
        Histogram-based gradient boosting model.
    """
    
    random_forest: RandomForestRegressor
    gradient_boosting: GradientBoostingRegressor
    hist_gradient_boosting: HistGradientBoostingRegressor


def train_tree_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[TreeModels, Dict[str, Dict[str, float]]]:
    """
    Train tree-based regression models (RandomForest, GradientBoosting,
    HistGradientBoosting) on the training data and evaluate them on train
    and validation sets.

    Hyperparameters are kept reasonably simple to avoid overfitting and
    excessive training time.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (imputed and scaled).
    y_train : np.ndarray
        Training targets.
    X_val : np.ndarray
        Validation feature matrix (imputed and scaled).
    y_val : np.ndarray
        Validation targets.

    Returns
    -------
    models : TreeModels
        Container with fitted sklearn models.
    metrics : dict
        Nested dict metrics[model_name][metric_name] = value.
    """
    logger.info("=" * 70)
    logger.info("TRAINING TREE-BASED MODELS")
    logger.info("=" * 70)
    
    # Initialize models with conservative hyperparameters
    models = TreeModels(
        random_forest=RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        ),
        gradient_boosting=GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42,
            verbose=0,
        ),
        hist_gradient_boosting=HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
            l2_regularization=0.0,
            verbose=0,
        ),
    )
    
    metrics: Dict[str, Dict[str, float]] = {}
    
    # Train and evaluate each model
    for name, model in [
        ("tree_rf", models.random_forest),
        ("tree_gbr", models.gradient_boosting),
        ("tree_hgb", models.hist_gradient_boosting),
    ]:
        model_display = {
            "tree_rf": "Random Forest",
            "tree_gbr": "Gradient Boosting",
            "tree_hgb": "Hist Gradient Boosting",
        }[name]
        
        logger.info(f"\nTraining {model_display}...")
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Generate predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Compute metrics
        metrics[name] = {
            **regression_metrics(y_train, y_train_pred, prefix="train_"),
            **regression_metrics(y_val, y_val_pred, prefix="val_"),
        }
        
        logger.info(f"  Train MAE: {metrics[name]['train_mae']:.6f}")
        logger.info(f"  Train RMSE: {metrics[name]['train_rmse']:.6f}")
        logger.info(f"  Train R²: {metrics[name]['train_r2']:.6f}")
        logger.info(f"  Val MAE: {metrics[name]['val_mae']:.6f}")
        logger.info(f"  Val RMSE: {metrics[name]['val_rmse']:.6f}")
        logger.info(f"  Val R²: {metrics[name]['val_r2']:.6f}")
        
        # Log feature importance for Random Forest and Gradient Boosting
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            n_important = np.sum(importances > 0.01)
            logger.info(f"  Features with importance >1%: {n_important} / {len(importances)}")
            logger.info(f"  Max feature importance: {importances.max():.4f}")
    
    return models, metrics
