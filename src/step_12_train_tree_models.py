"""
Step 12 — Tree-based regression models.

This step:
- Loads preprocessed train/val data from Step 10.
- Reuses the feature imputer fitted in Step 11.
- Trains tree-based regressors:
    * RandomForestRegressor
    * GradientBoostingRegressor
    * HistGradientBoostingRegressor
- Evaluates them on train and validation sets using MAE, RMSE, R².
- Merges their metrics into the existing regression_metrics_summary.json
  from Step 11, to allow direct comparison with baselines & linear models.
- Saves fitted tree models and an updated metrics summary.
"""

from typing import NoReturn, Dict
import json
import logging

import joblib
import numpy as np

from src.config import Settings
from src.models.tree_models import train_tree_models


def run_step_12() -> NoReturn:
    """
    Execute Step 12: Train tree-based regression models.
    
    This step loads preprocessed data from Step 10, applies the imputer
    from Step 11, trains tree-based models (Random Forest, Gradient Boosting,
    HistGradientBoosting), evaluates them on train and validation sets,
    and merges metrics with Step 11 results.
    """
    # Initialize settings
    settings = Settings()
    logger = settings.setup_logging("step_12_train_tree_models")
    
    logger.info("=" * 70)
    logger.info("STEP 12: TREE-BASED REGRESSION MODELS")
    logger.info("=" * 70)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Define paths
    step10_dir = settings.RESULTS_DIR / "step_10"
    step11_dir = settings.RESULTS_DIR / "step_11"
    output_dir = settings.RESULTS_DIR / "step_12"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nInput directories:")
    logger.info(f"  Step 10: {step10_dir}")
    logger.info(f"  Step 11: {step11_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # ========================================================================
    # Load preprocessed data from Step 10
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Loading preprocessed data from Step 10...")
    logger.info("-" * 70)
    
    # Load matrices
    X_train = np.load(step10_dir / "X_train.npy")
    X_val = np.load(step10_dir / "X_val.npy")
    y_train = np.load(step10_dir / "y_train.npy")
    y_val = np.load(step10_dir / "y_val.npy")
    
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  X_val shape: {X_val.shape}")
    logger.info(f"  y_train shape: {y_train.shape}")
    logger.info(f"  y_val shape: {y_val.shape}")
    
    # ========================================================================
    # Load and apply feature imputer from Step 11
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Loading and applying feature imputer from Step 11...")
    logger.info("-" * 70)
    
    imputer_path = step11_dir / "feature_imputer.joblib"
    if not imputer_path.exists():
        error_msg = (
            f"Feature imputer not found at {imputer_path}. "
            f"Please ensure Step 11 has been run successfully."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    imputer = joblib.load(imputer_path)
    logger.info(f"  ✓ Loaded feature imputer")
    
    # Apply imputation
    X_train_imp = imputer.transform(X_train)
    X_val_imp = imputer.transform(X_val)
    
    logger.info(f"  ✓ Applied imputation to train and validation sets")
    logger.info(f"  Imputed train shape: {X_train_imp.shape}")
    logger.info(f"  Imputed val shape: {X_val_imp.shape}")
    
    # ========================================================================
    # Train tree-based models
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Training tree-based models...")
    logger.info("-" * 70)
    
    tree_models, tree_metrics = train_tree_models(X_train_imp, y_train, X_val_imp, y_val)
    
    # ========================================================================
    # Load and merge metrics from Step 11
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Merging metrics with Step 11 results...")
    logger.info("-" * 70)
    
    metrics_path_step11 = step11_dir / "regression_metrics_summary.json"
    if metrics_path_step11.exists():
        with metrics_path_step11.open("r", encoding="utf-8") as f:
            metrics_summary = json.load(f)
        logger.info(f"  ✓ Loaded {len(metrics_summary)} models from Step 11")
    else:
        logger.warning(f"  Step 11 metrics not found at {metrics_path_step11}")
        metrics_summary = {}
    
    # Merge tree model metrics
    metrics_summary.update(tree_metrics)
    logger.info(f"  ✓ Added {len(tree_metrics)} tree models")
    logger.info(f"  Total models: {len(metrics_summary)}")
    
    # ========================================================================
    # Save models
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving models...")
    logger.info("-" * 70)
    
    joblib.dump(tree_models.random_forest, output_dir / "rf_model.joblib")
    logger.info(f"  ✓ Saved rf_model.joblib")
    
    joblib.dump(tree_models.gradient_boosting, output_dir / "gbr_model.joblib")
    logger.info(f"  ✓ Saved gbr_model.joblib")
    
    joblib.dump(tree_models.hist_gradient_boosting, output_dir / "hgb_model.joblib")
    logger.info(f"  ✓ Saved hgb_model.joblib")
    
    # ========================================================================
    # Save merged metrics
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving merged metrics...")
    logger.info("-" * 70)
    
    metrics_path_step12 = output_dir / "regression_metrics_summary_with_trees.json"
    with metrics_path_step12.open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"  ✓ Saved regression_metrics_summary_with_trees.json")
    
    # ========================================================================
    # Save completion marker
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving completion marker...")
    logger.info("-" * 70)
    
    marker_path = output_dir / "step_12_completed.txt"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write("Step 12 completed: tree-based regression models trained and evaluated on train/val.\n")
        f.write(f"Models trained: 3 (Random Forest, Gradient Boosting, Hist Gradient Boosting)\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Val samples: {len(y_val)}\n")
    
    logger.info(f"  ✓ Completion marker saved: {marker_path}")
    
    # ========================================================================
    # Summary and comparison
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 12 COMPLETED: MODEL COMPARISON")
    logger.info("=" * 70)
    
    logger.info("\nValidation MAE Comparison (All Models):")
    
    # Sort by validation MAE
    sorted_models = sorted(
        metrics_summary.items(),
        key=lambda x: x[1].get('val_mae', float('inf'))
    )
    
    logger.info(f"\n{'Model':<30} {'Val MAE':<12} {'Val R²':<12}")
    logger.info("-" * 70)
    
    model_display_names = {
        "baseline_mean": "Baseline (Mean)",
        "baseline_capm": "Baseline (CAPM)",
        "model_linear": "Linear (OLS)",
        "model_ridge": "Ridge",
        "model_lasso": "Lasso",
        "tree_rf": "Random Forest",
        "tree_gbr": "Gradient Boosting",
        "tree_hgb": "Hist Gradient Boosting",
    }
    
    for model_key, model_metrics in sorted_models:
        display_name = model_display_names.get(model_key, model_key)
        val_mae = model_metrics.get('val_mae', float('nan'))
        val_r2 = model_metrics.get('val_r2', float('nan'))
        logger.info(f"{display_name:<30} {val_mae:<12.6f} {val_r2:<12.6f}")
    
    logger.info("-" * 70)
    
    # Determine best model overall
    best_model_key = sorted_models[0][0]
    best_model_name = model_display_names.get(best_model_key, best_model_key)
    best_metrics = sorted_models[0][1]
    
    logger.info(f"\n🏆 Best model overall (lowest val MAE): {best_model_name}")
    logger.info(f"   Val MAE: {best_metrics['val_mae']:.6f}")
    logger.info(f"   Val R²: {best_metrics['val_r2']:.6f}")
    
    # Compare tree models to baselines
    if "baseline_mean" in metrics_summary:
        baseline_mae = metrics_summary["baseline_mean"]["val_mae"]
        best_tree_key = min(
            ["tree_rf", "tree_gbr", "tree_hgb"],
            key=lambda k: metrics_summary[k]["val_mae"]
        )
        best_tree_mae = metrics_summary[best_tree_key]["val_mae"]
        improvement = 100 * (baseline_mae - best_tree_mae) / baseline_mae
        
        logger.info(f"\n📊 Tree Model Performance:")
        logger.info(f"   Best tree model: {model_display_names[best_tree_key]}")
        logger.info(f"   Improvement over mean baseline: {improvement:.2f}%")
    
    logger.info("\n" + "=" * 70)
    logger.info("Key outputs:")
    logger.info(f"  - Tree models: rf_model.joblib, gbr_model.joblib, hgb_model.joblib")
    logger.info(f"  - Merged metrics: regression_metrics_summary_with_trees.json")
    
    logger.info("\nNext: Step 13 will evaluate all models on the test set.")


if __name__ == "__main__":
    run_step_12()
