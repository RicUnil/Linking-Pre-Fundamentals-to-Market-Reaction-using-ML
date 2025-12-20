"""
Step 13 — XGBoost regression models (advanced boosting).

This step:
- Loads preprocessed train/val data from Step 10.
- Reuses the feature imputer fitted in Step 11.
- Trains XGBoost regression models:
    * xgb_baseline
    * xgb_tuned
- Evaluates them on train and validation sets using MAE, RMSE, R².
- Merges their metrics into the existing regression metrics summary from
  previous steps (including baselines, linear models, and tree models).
- Saves fitted XGBoost models and an updated metrics summary file.

The main goal is to check whether a strong boosting model like XGBoost
can significantly outperform previous models, thereby challenging or
reinforcing H₀: excess returns are unpredictable.
"""

from typing import NoReturn, Dict
import json
import logging

import joblib
import numpy as np

from src.config import Settings
from src.models.xgb_models import train_xgb_models


def run_step_13() -> NoReturn:
    """
    Execute Step 13: Train XGBoost regression models.
    
    This step loads preprocessed data from Step 10, applies the imputer
    from Step 11, trains XGBoost models (baseline and tuned), evaluates
    them on train and validation sets, and merges metrics with previous
    steps' results.
    """
    # Initialize settings
    settings = Settings()
    logger = settings.setup_logging("step_13_train_xgboost_models")
    
    logger.info("=" * 70)
    logger.info("STEP 13: XGBOOST REGRESSION MODELS")
    logger.info("=" * 70)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Define paths
    step10_dir = settings.RESULTS_DIR / "step_10"
    step11_dir = settings.RESULTS_DIR / "step_11"
    step12_dir = settings.RESULTS_DIR / "step_12"
    output_dir = settings.RESULTS_DIR / "step_13"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nInput directories:")
    logger.info(f"  Step 10: {step10_dir}")
    logger.info(f"  Step 11: {step11_dir}")
    logger.info(f"  Step 12: {step12_dir}")
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
    # Train XGBoost models
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Training XGBoost models...")
    logger.info("-" * 70)
    
    xgb_models, xgb_metrics = train_xgb_models(X_train_imp, y_train, X_val_imp, y_val)
    
    # ========================================================================
    # Load and merge metrics from Step 12
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Merging metrics with previous results...")
    logger.info("-" * 70)
    
    metrics_path_step12 = step12_dir / "regression_metrics_summary_with_trees.json"
    if metrics_path_step12.exists():
        with metrics_path_step12.open("r", encoding="utf-8") as f:
            metrics_summary = json.load(f)
        logger.info(f"  ✓ Loaded {len(metrics_summary)} models from previous steps")
    else:
        logger.warning(f"  Step 12 metrics not found at {metrics_path_step12}")
        metrics_summary = {}
    
    # Merge XGBoost metrics
    metrics_summary.update(xgb_metrics)
    logger.info(f"  ✓ Added {len(xgb_metrics)} XGBoost models")
    logger.info(f"  Total models: {len(metrics_summary)}")
    
    # ========================================================================
    # Save models
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving models...")
    logger.info("-" * 70)
    
    joblib.dump(xgb_models.baseline, output_dir / "xgb_baseline_model.joblib")
    logger.info(f"  ✓ Saved xgb_baseline_model.joblib")
    
    joblib.dump(xgb_models.tuned, output_dir / "xgb_tuned_model.joblib")
    logger.info(f"  ✓ Saved xgb_tuned_model.joblib")
    
    # ========================================================================
    # Save validation predictions
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving validation predictions...")
    logger.info("-" * 70)
    
    # Save predictions from the tuned model
    val_pred = xgb_models.tuned.predict(X_val_imp)
    val_pred_path = output_dir / "val_predictions_xgb_tuned.npy"
    np.save(val_pred_path, val_pred)
    logger.info(f"  ✓ Saved val_predictions_xgb_tuned.npy")
    
    # ========================================================================
    # Save merged metrics
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving merged metrics...")
    logger.info("-" * 70)
    
    metrics_path_step13 = output_dir / "regression_metrics_summary_with_trees_and_xgb.json"
    with metrics_path_step13.open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"  ✓ Saved regression_metrics_summary_with_trees_and_xgb.json")
    
    # ========================================================================
    # Save completion marker
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving completion marker...")
    logger.info("-" * 70)
    
    marker_path = output_dir / "step_13_completed.txt"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write("Step 13 completed: XGBoost regression models trained and evaluated on train/val.\n")
        f.write(f"Models trained: 2 (XGBoost Baseline, XGBoost Tuned)\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Val samples: {len(y_val)}\n")
    
    logger.info(f"  ✓ Completion marker saved: {marker_path}")
    
    # ========================================================================
    # Summary and comparison
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 13 COMPLETED: MODEL COMPARISON")
    logger.info("=" * 70)
    
    logger.info("\nValidation MAE Comparison (All Models):")
    
    # Sort by validation MAE
    sorted_models = sorted(
        metrics_summary.items(),
        key=lambda x: x[1].get('val_mae', float('inf'))
    )
    
    logger.info(f"\n{'Model':<35} {'Val MAE':<12} {'Val R²':<12}")
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
        "xgb_baseline": "XGBoost Baseline (depth=3)",
        "xgb_tuned": "XGBoost Tuned (depth=5)",
    }
    
    for model_key, model_metrics in sorted_models:
        display_name = model_display_names.get(model_key, model_key)
        val_mae = model_metrics.get('val_mae', float('nan'))
        val_r2 = model_metrics.get('val_r2', float('nan'))
        logger.info(f"{display_name:<35} {val_mae:<12.6f} {val_r2:<12.6f}")
    
    logger.info("-" * 70)
    
    # Determine best model overall
    best_model_key = sorted_models[0][0]
    best_model_name = model_display_names.get(best_model_key, best_model_key)
    best_metrics = sorted_models[0][1]
    
    logger.info(f"\n🏆 Best model overall (lowest val MAE): {best_model_name}")
    logger.info(f"   Val MAE: {best_metrics['val_mae']:.6f}")
    logger.info(f"   Val R²: {best_metrics['val_r2']:.6f}")
    
    # Compare XGBoost to baselines
    if "baseline_mean" in metrics_summary:
        baseline_mae = metrics_summary["baseline_mean"]["val_mae"]
        best_xgb_key = min(
            ["xgb_baseline", "xgb_tuned"],
            key=lambda k: metrics_summary[k]["val_mae"]
        )
        best_xgb_mae = metrics_summary[best_xgb_key]["val_mae"]
        improvement = 100 * (baseline_mae - best_xgb_mae) / baseline_mae
        
        logger.info(f"\n📊 XGBoost Performance:")
        logger.info(f"   Best XGBoost model: {model_display_names[best_xgb_key]}")
        logger.info(f"   Improvement over mean baseline: {improvement:.2f}%")
        
        # Compare to best previous model
        if "model_ridge" in metrics_summary:
            ridge_mae = metrics_summary["model_ridge"]["val_mae"]
            xgb_vs_ridge = 100 * (ridge_mae - best_xgb_mae) / ridge_mae
            logger.info(f"   Improvement over Ridge: {xgb_vs_ridge:.2f}%")
    
    logger.info("\n" + "=" * 70)
    logger.info("Key outputs:")
    logger.info(f"  - XGBoost models: xgb_baseline_model.joblib, xgb_tuned_model.joblib")
    logger.info(f"  - Merged metrics: regression_metrics_summary_with_trees_and_xgb.json")
    logger.info(f"  - Predictions: val_predictions_xgb_tuned.npy")
    
    logger.info("\nNext: Step 14 will likely evaluate all models on the test set.")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_step_13()
