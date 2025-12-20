"""
Step 11 — Baseline and linear regression models (testing H₀).

This step:
- Loads preprocessed train/val/test data from Step 10.
- Trains:
    * Dummy mean baseline
    * CAPM-style baseline (per-ticker beta using SPY pre-return)
    * LinearRegression
    * Ridge
    * Lasso
- Evaluates all models on train and validation sets using MAE, RMSE, R².
- Saves fitted models, metrics summary, and a clear comparison vs baselines.

The main goal is to start testing H₀: 30-day post-earnings excess returns
are unpredictable, by checking whether ML models significantly outperform
strong baselines on the validation set.
"""

from typing import NoReturn, Dict
import json
import logging

import joblib
import numpy as np
import pandas as pd

from src.config import Settings
from src.models.baselines import train_baseline_models
from src.models.linear_models import train_linear_models
from src.metrics.regression import regression_metrics


def run_step_11() -> NoReturn:
    """
    Execute Step 11: Train baseline and linear regression models.
    
    This step loads preprocessed data from Step 10, trains baseline models
    (mean and CAPM) and linear models (OLS, Ridge, Lasso), evaluates them
    on train and validation sets, and saves all artifacts.
    """
    # Initialize settings
    settings = Settings()
    logger = settings.setup_logging("step_11_train_baseline_and_linear_models")
    
    logger.info("=" * 70)
    logger.info("STEP 11: BASELINE AND LINEAR REGRESSION MODELS")
    logger.info("=" * 70)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Define paths
    base_dir = settings.RESULTS_DIR / "step_10"
    output_dir = settings.RESULTS_DIR / "step_11"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nInput directory: {base_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # ========================================================================
    # Load preprocessed data from Step 10
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Loading preprocessed data from Step 10...")
    logger.info("-" * 70)
    
    # Load matrices
    X_train = np.load(base_dir / "X_train.npy")
    X_val = np.load(base_dir / "X_val.npy")
    y_train = np.load(base_dir / "y_train.npy")
    y_val = np.load(base_dir / "y_val.npy")
    
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  X_val shape: {X_val.shape}")
    logger.info(f"  y_train shape: {y_train.shape}")
    logger.info(f"  y_val shape: {y_val.shape}")
    
    # Handle missing values with simple imputation (median)
    from sklearn.impute import SimpleImputer
    
    logger.info("\n  Handling missing values...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    n_missing_train = np.isnan(X_train).sum()
    n_missing_val = np.isnan(X_val).sum()
    logger.info(f"  Missing values in train: {n_missing_train:,} ({100*n_missing_train/X_train.size:.1f}%)")
    logger.info(f"  Missing values in val: {n_missing_val:,} ({100*n_missing_val/X_val.size:.1f}%)")
    logger.info(f"  Imputation strategy: median")
    
    # Save imputer for later use
    joblib.dump(imputer, output_dir / "feature_imputer.joblib")
    logger.info(f"  ✓ Saved feature_imputer.joblib")
    
    # Load cleaned dataframes (needed for CAPM baseline)
    df_train = pd.read_parquet(base_dir / "cleaned_train.parquet")
    df_val = pd.read_parquet(base_dir / "cleaned_val.parquet")
    
    logger.info(f"  Loaded cleaned dataframes for CAPM baseline")
    
    # ========================================================================
    # Train baseline models
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Training baseline models...")
    logger.info("-" * 70)
    
    baseline_results = train_baseline_models(df_train, df_val, y_train, y_val, settings)
    
    # Compute metrics for baselines
    metrics_summary: Dict[str, Dict[str, float]] = {}
    
    logger.info("\nComputing baseline metrics...")
    
    metrics_summary["baseline_mean"] = {
        **regression_metrics(y_train, baseline_results.mean_train_pred, prefix="train_"),
        **regression_metrics(y_val, baseline_results.mean_val_pred, prefix="val_"),
    }
    
    metrics_summary["baseline_capm"] = {
        **regression_metrics(y_train, baseline_results.capm_train_pred, prefix="train_"),
        **regression_metrics(y_val, baseline_results.capm_val_pred, prefix="val_"),
    }
    
    logger.info("\nBaseline Mean:")
    logger.info(f"  Train MAE: {metrics_summary['baseline_mean']['train_mae']:.6f}")
    logger.info(f"  Val MAE: {metrics_summary['baseline_mean']['val_mae']:.6f}")
    logger.info(f"  Val R²: {metrics_summary['baseline_mean']['val_r2']:.6f}")
    
    logger.info("\nBaseline CAPM:")
    logger.info(f"  Train MAE: {metrics_summary['baseline_capm']['train_mae']:.6f}")
    logger.info(f"  Val MAE: {metrics_summary['baseline_capm']['val_mae']:.6f}")
    logger.info(f"  Val R²: {metrics_summary['baseline_capm']['val_r2']:.6f}")
    
    # ========================================================================
    # Train linear models
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Training linear models...")
    logger.info("-" * 70)
    
    linear_models, linear_metrics = train_linear_models(X_train_imputed, y_train, X_val_imputed, y_val)
    
    # Merge metrics with 'model_' prefix for clarity
    metrics_summary.update(
        {f"model_{name}": m for name, m in linear_metrics.items()}
    )
    
    # ========================================================================
    # Save models
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving models...")
    logger.info("-" * 70)
    
    # Save baseline models
    joblib.dump(
        baseline_results.mean_model,
        output_dir / "baseline_mean_model.joblib",
    )
    logger.info(f"  ✓ Saved baseline_mean_model.joblib")
    
    baseline_results.capm_betas.to_parquet(output_dir / "baseline_capm_betas.parquet")
    logger.info(f"  ✓ Saved baseline_capm_betas.parquet ({len(baseline_results.capm_betas)} tickers)")
    
    # Save linear models
    joblib.dump(linear_models.linear, output_dir / "linear_model.joblib")
    logger.info(f"  ✓ Saved linear_model.joblib")
    
    joblib.dump(linear_models.ridge, output_dir / "ridge_model.joblib")
    logger.info(f"  ✓ Saved ridge_model.joblib")
    
    joblib.dump(linear_models.lasso, output_dir / "lasso_model.joblib")
    logger.info(f"  ✓ Saved lasso_model.joblib")
    
    # ========================================================================
    # Save metrics
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving metrics...")
    logger.info("-" * 70)
    
    metrics_path = output_dir / "regression_metrics_summary.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"  ✓ Saved regression_metrics_summary.json")
    
    # Save validation predictions for later analysis
    val_preds = {
        "baseline_mean": baseline_results.mean_val_pred.tolist(),
        "baseline_capm": baseline_results.capm_val_pred.tolist(),
    }
    pred_path = output_dir / "val_predictions_baselines.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(val_preds, f)
    logger.info(f"  ✓ Saved val_predictions_baselines.json")
    
    # ========================================================================
    # Save completion marker
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving completion marker...")
    logger.info("-" * 70)
    
    marker_path = output_dir / "step_11_completed.txt"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write("Step 11 completed: baselines and linear models trained.\n")
        f.write(f"Models trained: 5 (2 baselines + 3 linear)\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Val samples: {len(y_val)}\n")
    
    logger.info(f"  ✓ Completion marker saved: {marker_path}")
    
    # ========================================================================
    # Summary and comparison
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 11 COMPLETED: MODEL COMPARISON")
    logger.info("=" * 70)
    
    logger.info("\nValidation MAE Comparison:")
    logger.info(f"  Baseline Mean:  {metrics_summary['baseline_mean']['val_mae']:.6f}")
    logger.info(f"  Baseline CAPM:  {metrics_summary['baseline_capm']['val_mae']:.6f}")
    logger.info(f"  Linear (OLS):   {metrics_summary['model_linear']['val_mae']:.6f}")
    logger.info(f"  Ridge:          {metrics_summary['model_ridge']['val_mae']:.6f}")
    logger.info(f"  Lasso:          {metrics_summary['model_lasso']['val_mae']:.6f}")
    
    logger.info("\nValidation R² Comparison:")
    logger.info(f"  Baseline Mean:  {metrics_summary['baseline_mean']['val_r2']:.6f}")
    logger.info(f"  Baseline CAPM:  {metrics_summary['baseline_capm']['val_r2']:.6f}")
    logger.info(f"  Linear (OLS):   {metrics_summary['model_linear']['val_r2']:.6f}")
    logger.info(f"  Ridge:          {metrics_summary['model_ridge']['val_r2']:.6f}")
    logger.info(f"  Lasso:          {metrics_summary['model_lasso']['val_r2']:.6f}")
    
    # Determine best model
    best_model = min(
        ["baseline_mean", "baseline_capm", "model_linear", "model_ridge", "model_lasso"],
        key=lambda m: metrics_summary[m]["val_mae"]
    )
    
    logger.info(f"\n🏆 Best model (lowest val MAE): {best_model}")
    logger.info(f"   Val MAE: {metrics_summary[best_model]['val_mae']:.6f}")
    logger.info(f"   Val R²: {metrics_summary[best_model]['val_r2']:.6f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Key outputs:")
    logger.info(f"  - Baseline models: baseline_mean_model.joblib, baseline_capm_betas.parquet")
    logger.info(f"  - Linear models: linear_model.joblib, ridge_model.joblib, lasso_model.joblib")
    logger.info(f"  - Metrics: regression_metrics_summary.json")
    logger.info(f"  - Predictions: val_predictions_baselines.json")
    
    logger.info("\nNext: Step 12 will implement tree-based models and further test H₀.")


if __name__ == "__main__":
    run_step_11()
