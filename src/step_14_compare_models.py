"""
Step 14 — Full Model Comparison (Train + Validation)

This step:
- Loads all trained regression models from Steps 11–13.
- Loads imputed features (X_train, X_val) and targets (y_train, y_val).
- Recomputes predictions consistently for all 10 models.
- Evaluates with regression_metrics() → MAE, RMSE, R².
- Produces:
    * model_comparison.json
    * model_comparison.csv
    * ranked_models.json (sorted by validation MAE)
- No test set evaluation yet (reserved for Step 18).
- No visualizations yet (reserved for Step 19).
"""

from typing import NoReturn, Dict, Any
from pathlib import Path
import numpy as np
import joblib
import json
import pandas as pd
import logging

from src.config import Settings
from src.metrics.regression import regression_metrics


def load_all_models(settings: Settings) -> Dict[str, Any]:
    """
    Load all trained regression models from Steps 11-13.
    
    Parameters
    ----------
    settings : Settings
        Project settings with directory paths.
    
    Returns
    -------
    models : dict
        Dictionary mapping model names to loaded model objects.
        Special handling for CAPM baseline (returns betas DataFrame).
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading all trained models...")
    
    models = {}
    
    # Step 11: Baselines and Linear models
    step11_dir = settings.RESULTS_DIR / "step_11"
    
    # Mean baseline
    models["baseline_mean"] = joblib.load(step11_dir / "baseline_mean_model.joblib")
    logger.info("  ✓ Loaded baseline_mean")
    
    # CAPM baseline (betas)
    models["baseline_capm"] = pd.read_parquet(step11_dir / "baseline_capm_betas.parquet")
    logger.info("  ✓ Loaded baseline_capm (betas)")
    
    # Linear models
    models["linear_ols"] = joblib.load(step11_dir / "linear_model.joblib")
    logger.info("  ✓ Loaded linear_ols")
    
    models["ridge"] = joblib.load(step11_dir / "ridge_model.joblib")
    logger.info("  ✓ Loaded ridge")
    
    models["lasso"] = joblib.load(step11_dir / "lasso_model.joblib")
    logger.info("  ✓ Loaded lasso")
    
    # Step 12: Tree models
    step12_dir = settings.RESULTS_DIR / "step_12"
    
    models["random_forest"] = joblib.load(step12_dir / "rf_model.joblib")
    logger.info("  ✓ Loaded random_forest")
    
    models["gradient_boosting"] = joblib.load(step12_dir / "gbr_model.joblib")
    logger.info("  ✓ Loaded gradient_boosting")
    
    models["hist_gradient_boosting"] = joblib.load(step12_dir / "hgb_model.joblib")
    logger.info("  ✓ Loaded hist_gradient_boosting")
    
    # Step 13: XGBoost-style models (HistGB fallback)
    step13_dir = settings.RESULTS_DIR / "step_13"
    
    models["xgb_baseline"] = joblib.load(step13_dir / "xgb_baseline_model.joblib")
    logger.info("  ✓ Loaded xgb_baseline")
    
    models["xgb_tuned"] = joblib.load(step13_dir / "xgb_tuned_model.joblib")
    logger.info("  ✓ Loaded xgb_tuned")
    
    logger.info(f"Total models loaded: {len(models)}")
    
    return models


def predict_capm_baseline(
    betas_df: pd.DataFrame,
    df_clean: pd.DataFrame,
    y_indices: np.ndarray,
    global_mean: float,
) -> np.ndarray:
    """
    Generate CAPM baseline predictions.
    
    Parameters
    ----------
    betas_df : pd.DataFrame
        DataFrame with index='ticker' and columns ['beta', 'alpha'].
    df_clean : pd.DataFrame
        Cleaned dataframe with 'ticker' and 'spy_pre_return_30d' columns.
    y_indices : np.ndarray
        Indices to select from df_clean.
    global_mean : float
        Global mean to use as fallback.
    
    Returns
    -------
    predictions : np.ndarray
        CAPM predictions.
    """
    predictions = np.full(len(y_indices), global_mean)
    
    # Create a mapping from ticker to beta (ticker is the index)
    beta_dict = betas_df['beta'].to_dict()
    
    # Get subset of data
    df_subset = df_clean.iloc[y_indices].copy()
    
    for i, (idx, row) in enumerate(df_subset.iterrows()):
        ticker = row['ticker']
        spy_pre = row.get('spy_pre_return_30d', np.nan)
        
        if ticker in beta_dict and not pd.isna(spy_pre):
            beta = beta_dict[ticker]
            predictions[i] = beta * spy_pre
        # else: keep global mean
    
    return predictions


def run_step_14() -> NoReturn:
    """
    Execute Step 14: Compare all models on train and validation sets.
    
    This step loads all trained models, generates predictions on train
    and validation sets, computes metrics, and creates comparison tables.
    """
    # Initialize settings
    settings = Settings()
    logger = settings.setup_logging("step_14_compare_models")
    
    logger.info("=" * 70)
    logger.info("STEP 14: FULL MODEL COMPARISON (TRAIN + VALIDATION)")
    logger.info("=" * 70)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Define paths
    step10_dir = settings.RESULTS_DIR / "step_10"
    step11_dir = settings.RESULTS_DIR / "step_11"
    output_dir = settings.RESULTS_DIR / "step_14"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nOutput directory: {output_dir}")
    
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
    
    # Load cleaned dataframes for CAPM baseline
    df_train = pd.read_parquet(step10_dir / "cleaned_train.parquet")
    df_val = pd.read_parquet(step10_dir / "cleaned_val.parquet")
    
    logger.info(f"  Loaded cleaned dataframes for CAPM baseline")
    
    # Load feature imputer from Step 11
    imputer_path = step11_dir / "feature_imputer.joblib"
    if not imputer_path.exists():
        error_msg = f"Feature imputer not found at {imputer_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    imputer = joblib.load(imputer_path)
    logger.info(f"  ✓ Loaded feature imputer")
    
    # Apply imputation
    X_train_imp = imputer.transform(X_train)
    X_val_imp = imputer.transform(X_val)
    
    logger.info(f"  ✓ Applied imputation")
    
    # ========================================================================
    # Load all models
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Loading all models...")
    logger.info("-" * 70)
    
    models = load_all_models(settings)
    
    # ========================================================================
    # Generate predictions and compute metrics
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Generating predictions and computing metrics...")
    logger.info("-" * 70)
    
    all_metrics = {}
    
    # Define model categories
    model_categories = {
        "baseline_mean": "Baseline",
        "baseline_capm": "Baseline",
        "linear_ols": "Linear",
        "ridge": "Linear",
        "lasso": "Linear",
        "random_forest": "Tree",
        "gradient_boosting": "Tree",
        "hist_gradient_boosting": "Tree",
        "xgb_baseline": "Boosting",
        "xgb_tuned": "Boosting",
    }
    
    for model_name in models.keys():
        logger.info(f"\nEvaluating {model_name}...")
        
        model = models[model_name]
        
        # Special handling for CAPM baseline
        if model_name == "baseline_capm":
            # CAPM uses betas and SPY pre-return
            global_mean = y_train.mean()
            
            y_train_pred = predict_capm_baseline(
                model, df_train, np.arange(len(y_train)), global_mean
            )
            y_val_pred = predict_capm_baseline(
                model, df_val, np.arange(len(y_val)), global_mean
            )
        else:
            # Standard sklearn-style models
            y_train_pred = model.predict(X_train_imp)
            y_val_pred = model.predict(X_val_imp)
        
        # Compute metrics
        all_metrics[model_name] = {
            "category": model_categories[model_name],
            **regression_metrics(y_train, y_train_pred, prefix="train_"),
            **regression_metrics(y_val, y_val_pred, prefix="val_"),
        }
        
        logger.info(f"  Train MAE: {all_metrics[model_name]['train_mae']:.6f}")
        logger.info(f"  Val MAE: {all_metrics[model_name]['val_mae']:.6f}")
        logger.info(f"  Val R²: {all_metrics[model_name]['val_r2']:.6f}")
    
    # ========================================================================
    # Create comparison DataFrame
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Creating comparison tables...")
    logger.info("-" * 70)
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    comparison_df.index.name = 'model_name'
    comparison_df = comparison_df.reset_index()
    
    # Reorder columns
    column_order = [
        'model_name', 'category',
        'train_mae', 'train_rmse', 'train_r2',
        'val_mae', 'val_rmse', 'val_r2'
    ]
    comparison_df = comparison_df[column_order]
    
    # Sort by validation MAE
    comparison_df_sorted = comparison_df.sort_values('val_mae').reset_index(drop=True)
    
    logger.info(f"  ✓ Created comparison table with {len(comparison_df)} models")
    
    # ========================================================================
    # Create ranked models list
    # ========================================================================
    
    ranked_models = []
    for rank, (idx, row) in enumerate(comparison_df_sorted.iterrows(), 1):
        ranked_models.append({
            "rank": rank,
            "model_name": row['model_name'],
            "category": row['category'],
            "val_mae": float(row['val_mae']),
            "val_rmse": float(row['val_rmse']),
            "val_r2": float(row['val_r2']),
            "train_mae": float(row['train_mae']),
            "train_rmse": float(row['train_rmse']),
            "train_r2": float(row['train_r2']),
        })
    
    # ========================================================================
    # Save results
    # ========================================================================
    
    logger.info("\n" + "-" * 70)
    logger.info("Saving results...")
    logger.info("-" * 70)
    
    # Save CSV
    csv_path = output_dir / "model_comparison.csv"
    comparison_df_sorted.to_csv(csv_path, index=False)
    logger.info(f"  ✓ Saved {csv_path.name}")
    
    # Save JSON (full comparison)
    json_path = output_dir / "model_comparison.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"  ✓ Saved {json_path.name}")
    
    # Save ranked models JSON
    ranked_path = output_dir / "ranked_models.json"
    with ranked_path.open("w", encoding="utf-8") as f:
        json.dump(ranked_models, f, indent=2)
    logger.info(f"  ✓ Saved {ranked_path.name}")
    
    # Save completion marker
    marker_path = output_dir / "step_14_completed.txt"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write("Step 14 completed: all models evaluated on train/validation and ranked.\n")
        f.write(f"Total models compared: {len(all_metrics)}\n")
        f.write(f"Best model (val MAE): {ranked_models[0]['model_name']}\n")
        f.write(f"Best val MAE: {ranked_models[0]['val_mae']:.6f}\n")
    
    logger.info(f"  ✓ Saved {marker_path.name}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 14 COMPLETED: MODEL COMPARISON SUMMARY")
    logger.info("=" * 70)
    
    logger.info("\nTop 5 Models (by Validation MAE):")
    logger.info("-" * 70)
    logger.info(f"{'Rank':<6} {'Model':<30} {'Val MAE':<12} {'Val R²':<12}")
    logger.info("-" * 70)
    
    for model in ranked_models[:5]:
        logger.info(
            f"{model['rank']:<6} {model['model_name']:<30} "
            f"{model['val_mae']:<12.6f} {model['val_r2']:<12.6f}"
        )
    
    logger.info("-" * 70)
    
    # Best model details
    best = ranked_models[0]
    logger.info(f"\n🏆 Best Model: {best['model_name']}")
    logger.info(f"   Category: {best['category']}")
    logger.info(f"   Val MAE: {best['val_mae']:.6f}")
    logger.info(f"   Val RMSE: {best['val_rmse']:.6f}")
    logger.info(f"   Val R²: {best['val_r2']:.6f}")
    
    # Baseline comparison
    baseline_mean_metrics = next(m for m in ranked_models if m['model_name'] == 'baseline_mean')
    improvement = 100 * (baseline_mean_metrics['val_mae'] - best['val_mae']) / baseline_mean_metrics['val_mae']
    
    logger.info(f"\n📊 Performance vs Baseline:")
    logger.info(f"   Baseline (Mean) Val MAE: {baseline_mean_metrics['val_mae']:.6f}")
    logger.info(f"   Best Model Val MAE: {best['val_mae']:.6f}")
    logger.info(f"   Improvement: {improvement:.2f}%")
    
    logger.info("\n" + "=" * 70)
    logger.info("Key outputs:")
    logger.info(f"  - model_comparison.csv")
    logger.info(f"  - model_comparison.json")
    logger.info(f"  - ranked_models.json")
    
    logger.info("\nNext: Step 15+ will evaluate on test set and create visualizations.")
    logger.info("=" * 70)
    
    print("\nStep 14 complete: all models evaluated on train/validation and ranked.")


if __name__ == "__main__":
    run_step_14()
