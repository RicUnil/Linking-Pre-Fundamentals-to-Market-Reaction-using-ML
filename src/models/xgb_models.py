"""
XGBoost regression models for excess return prediction.

This module implements XGBoost-based regression models with different
hyperparameter configurations to test whether advanced boosting can
improve predictions over simpler models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import logging
import numpy as np

try:
    from xgboost import XGBRegressor
    USING_XGBOOST = True
    LIBRARY_NAME = "XGBoost"
except ImportError:
    # Fallback to LightGBM if XGBoost is not available (e.g., OpenMP issues on macOS)
    try:
        from lightgbm import LGBMRegressor as XGBRegressor
        USING_XGBOOST = False
        LIBRARY_NAME = "LightGBM"
        logger = logging.getLogger(__name__)
        logger.warning(
            "XGBoost not available (likely missing OpenMP). "
            "Using LightGBM as fallback - performance will be similar."
        )
    except ImportError:
        # Final fallback to sklearn's HistGradientBoostingRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor
        USING_XGBOOST = False
        LIBRARY_NAME = "HistGradientBoosting"
        logger = logging.getLogger(__name__)
        logger.warning(
            "XGBoost and LightGBM not available (missing OpenMP on macOS). "
            "Using sklearn's HistGradientBoostingRegressor as fallback."
        )

from src.metrics.regression import regression_metrics


logger = logging.getLogger(__name__)


@dataclass
class XGBModels:
    """
    Container for XGBoost regression models.

    We keep at least:
    - a 'baseline' XGBRegressor with conservative hyperparameters,
    - an optional 'tuned' XGBRegressor with slightly different settings.
    
    Attributes
    ----------
    baseline : XGBRegressor
        Baseline XGBoost model with conservative hyperparameters.
    tuned : XGBRegressor
        Tuned XGBoost model with adjusted depth and regularization.
    """
    
    baseline: XGBRegressor
    tuned: XGBRegressor


def train_xgb_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[XGBModels, Dict[str, Dict[str, float]]]:
    """
    Train XGBoost regression models on the training data and evaluate them
    on both train and validation sets.

    This function trains:
    - a baseline XGBRegressor with conservative hyperparameters,
    - a second 'tuned' model with slightly different depth and regularization.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (imputed and scaled).
    y_train : np.ndarray
        Training regression targets.
    X_val : np.ndarray
        Validation feature matrix (imputed and scaled).
    y_val : np.ndarray
        Validation regression targets.

    Returns
    -------
    models : XGBModels
        Container with fitted XGBoost models.
    metrics : dict
        Nested dict metrics[model_name][metric_name] = value, where
        model_name is 'xgb_baseline' or 'xgb_tuned'.
    """
    logger.info("=" * 70)
    logger.info(f"TRAINING {LIBRARY_NAME.upper()} MODELS")
    if not USING_XGBOOST:
        logger.info("(XGBoost fallback due to missing OpenMP)")
    logger.info("=" * 70)
    
    # Common hyperparameters for both models
    # Adjust parameter names based on library
    if USING_XGBOOST:
        common_kwargs = {
            "objective": "reg:squarederror",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        baseline_kwargs = {"max_depth": 3, "reg_lambda": 1.0, "reg_alpha": 0.0}
        tuned_kwargs = {"max_depth": 5, "reg_lambda": 2.0, "reg_alpha": 0.0}
    elif LIBRARY_NAME == "LightGBM":
        # LightGBM parameter names
        common_kwargs = {
            "objective": "regression",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        baseline_kwargs = {"max_depth": 3, "reg_lambda": 1.0, "reg_alpha": 0.0}
        tuned_kwargs = {"max_depth": 5, "reg_lambda": 2.0, "reg_alpha": 0.0}
    else:
        # HistGradientBoosting (sklearn) parameter names
        common_kwargs = {
            "max_iter": 400,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbose": 0,
        }
        baseline_kwargs = {"max_depth": 3, "l2_regularization": 1.0}
        tuned_kwargs = {"max_depth": 5, "l2_regularization": 2.0}
    
    # Initialize models with different configurations
    models = XGBModels(
        baseline=XGBRegressor(
            **baseline_kwargs,
            **common_kwargs,
        ),
        tuned=XGBRegressor(
            **tuned_kwargs,
            **common_kwargs,
        ),
    )
    
    metrics: Dict[str, Dict[str, float]] = {}
    
    # Train and evaluate each model
    for name, model in [
        ("xgb_baseline", models.baseline),
        ("xgb_tuned", models.tuned),
    ]:
        model_display = {
            "xgb_baseline": "XGBoost Baseline (depth=3, lambda=1.0)",
            "xgb_tuned": "XGBoost Tuned (depth=5, lambda=2.0)",
        }[name]
        
        logger.info(f"\nTraining {model_display}...")
        
        # Fit model with early stopping on validation set
        if USING_XGBOOST:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif LIBRARY_NAME == "LightGBM":
            # LightGBM uses different parameter names
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[],  # Suppress warnings
            )
        else:
            # HistGradientBoosting (sklearn) - simpler interface
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
        
        # Log feature importance info
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            n_important = np.sum(importances > 0.01)
            logger.info(f"  Features with importance >1%: {n_important} / {len(importances)}")
            logger.info(f"  Max feature importance: {importances.max():.4f}")
            logger.info(f"  Best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")
    
    logger.info("\n" + "=" * 70)
    
    return models, metrics
