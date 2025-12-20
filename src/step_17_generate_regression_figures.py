"""
Step 17 — Regression Visualizations & PNG Figures

This step:
- Loads trained models and prepared data from previous steps.
- Regenerates predictions for selected models on validation and test sets.
- Uses residuals and rolling metrics to create multiple PNG figures:
    * Actual vs predicted (val/test)
    * Residual histograms
    * Residuals vs predicted
    * Bar charts of MAE/R² by model (validation and test)
    * Rolling-window MAE and R² over time (per model)
- Saves all plots under results/step_17/figures/.

This step is purely about regression visualizations; it does not retrain
any model and does not change previous results.
"""

from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd

from src.config import Settings
from src.visualization.regression_plots import (
    plot_actual_vs_predicted_scatter,
    plot_residuals_histogram,
    plot_residuals_vs_predictions,
    plot_bar_metrics_by_model,
    plot_rolling_metric_over_time,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_step_17() -> None:
    """
    Execute Step 17: Generate regression visualization figures.

    This function:
    1. Loads validation and test data from Step 10
    2. Loads trained models from Steps 11-13
    3. Generates predictions for key models
    4. Creates visualization figures:
       - Actual vs predicted scatter plots
       - Residual histograms and scatter plots
       - Bar charts of metrics by model
       - Rolling-window performance over time
    5. Saves all figures as PNG files
    """
    logger.info("=" * 70)
    logger.info("STEP 17: REGRESSION VISUALIZATIONS & PNG FIGURES")
    logger.info("=" * 70)
    
    settings = Settings()
    settings.ensure_directories()

    results_dir = settings.RESULTS_DIR
    step10_dir = results_dir / "step_10"
    step11_dir = results_dir / "step_11"
    step12_dir = results_dir / "step_12"
    step13_dir = results_dir / "step_13"
    step14_dir = results_dir / "step_14"
    step15_dir = results_dir / "step_15"
    step16_dir = results_dir / "step_16"

    output_dir = results_dir / "step_17" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # =========================================================================
    # 1) LOAD VALIDATION AND TEST DATA
    # =========================================================================
    logger.info("Loading validation and test data from Step 10...")
    
    X_val = np.load(step10_dir / "X_val.npy")
    y_val = np.load(step10_dir / "y_val.npy")
    X_test = np.load(step10_dir / "X_test.npy")
    y_test = np.load(step10_dir / "y_test.npy")
    
    logger.info(f"  Validation set: {X_val.shape[0]} observations")
    logger.info(f"  Test set: {X_test.shape[0]} observations")

    # Use the same imputer from Step 11 for val/test predictions
    logger.info("Loading feature imputer from Step 11...")
    imputer = joblib.load(step11_dir / "feature_imputer.joblib")
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    logger.info("  ✓ Features imputed")
    logger.info("")

    # =========================================================================
    # 2) LOAD MODELS
    # =========================================================================
    logger.info("Loading trained models...")
    
    baseline_mean = joblib.load(step11_dir / "baseline_mean_model.joblib")
    logger.info("  ✓ Baseline mean model loaded")
    
    ridge_model = joblib.load(step11_dir / "ridge_model.joblib")
    logger.info("  ✓ Ridge model loaded")
    
    rf_model = joblib.load(step12_dir / "rf_model.joblib")
    logger.info("  ✓ Random Forest model loaded")
    
    # Use xgb_baseline as determined in Step 15
    xgb_best = joblib.load(step13_dir / "xgb_baseline_model.joblib")
    logger.info("  ✓ XGB best model loaded")
    logger.info("")

    # =========================================================================
    # 3) GENERATE PREDICTIONS
    # =========================================================================
    logger.info("Generating predictions on validation and test sets...")
    
    # Validation predictions
    y_val_pred_mean = baseline_mean.predict(np.zeros((X_val_imp.shape[0], 1)))
    y_val_pred_ridge = ridge_model.predict(X_val_imp)
    y_val_pred_rf = rf_model.predict(X_val_imp)
    y_val_pred_xgb = xgb_best.predict(X_val_imp)
    
    logger.info("  ✓ Validation predictions generated")

    # Test predictions
    y_test_pred_mean = baseline_mean.predict(np.zeros((X_test_imp.shape[0], 1)))
    y_test_pred_ridge = ridge_model.predict(X_test_imp)
    y_test_pred_rf = rf_model.predict(X_test_imp)
    y_test_pred_xgb = xgb_best.predict(X_test_imp)
    
    logger.info("  ✓ Test predictions generated")

    # Ridge residuals on test
    residuals_ridge_test = y_test - y_test_pred_ridge
    logger.info(f"  ✓ Ridge test residuals computed (mean: {residuals_ridge_test.mean():.6f})")
    logger.info("")

    # =========================================================================
    # 4) GENERATE FIGURES
    # =========================================================================
    logger.info("=" * 70)
    logger.info("GENERATING VISUALIZATION FIGURES")
    logger.info("=" * 70)
    logger.info("")

    # -------------------------------------------------------------------------
    # A) ACTUAL VS PREDICTED SCATTER PLOTS
    # -------------------------------------------------------------------------
    logger.info("Creating actual vs predicted scatter plots...")
    
    plot_actual_vs_predicted_scatter(
        y_true=y_val,
        y_pred=y_val_pred_ridge,
        out_path=output_dir / "actual_vs_pred_ridge_val.png",
        title="Ridge — Actual vs Predicted (Validation)",
    )
    logger.info("  ✓ actual_vs_pred_ridge_val.png")

    plot_actual_vs_predicted_scatter(
        y_true=y_test,
        y_pred=y_test_pred_ridge,
        out_path=output_dir / "actual_vs_pred_ridge_test.png",
        title="Ridge — Actual vs Predicted (Test)",
    )
    logger.info("  ✓ actual_vs_pred_ridge_test.png")
    
    # Also create for Random Forest (best on test in Step 15)
    plot_actual_vs_predicted_scatter(
        y_true=y_test,
        y_pred=y_test_pred_rf,
        out_path=output_dir / "actual_vs_pred_rf_test.png",
        title="Random Forest — Actual vs Predicted (Test)",
    )
    logger.info("  ✓ actual_vs_pred_rf_test.png")
    logger.info("")

    # -------------------------------------------------------------------------
    # B) RESIDUAL PLOTS
    # -------------------------------------------------------------------------
    logger.info("Creating residual plots...")
    
    plot_residuals_histogram(
        residuals=residuals_ridge_test,
        out_path=output_dir / "residuals_hist_ridge_test.png",
        title="Ridge — Residuals Histogram (Test)",
    )
    logger.info("  ✓ residuals_hist_ridge_test.png")

    plot_residuals_vs_predictions(
        y_pred=y_test_pred_ridge,
        residuals=residuals_ridge_test,
        out_path=output_dir / "residuals_vs_pred_ridge_test.png",
        title="Ridge — Residuals vs Predicted (Test)",
    )
    logger.info("  ✓ residuals_vs_pred_ridge_test.png")
    logger.info("")

    # -------------------------------------------------------------------------
    # C) BAR CHARTS OF METRICS BY MODEL
    # -------------------------------------------------------------------------
    logger.info("Creating bar charts of metrics by model...")
    
    # Load validation metrics from Step 14
    df_val_models = pd.read_csv(step14_dir / "model_comparison.csv")
    
    plot_bar_metrics_by_model(
        df_metrics=df_val_models,
        metric_col="val_mae",
        out_path=output_dir / "val_mae_by_model.png",
        title="Validation MAE by Model",
        ylabel="Validation MAE",
    )
    logger.info("  ✓ val_mae_by_model.png")
    
    plot_bar_metrics_by_model(
        df_metrics=df_val_models,
        metric_col="val_r2",
        out_path=output_dir / "val_r2_by_model.png",
        title="Validation R² by Model",
        ylabel="Validation R²",
    )
    logger.info("  ✓ val_r2_by_model.png")

    # Load test metrics from Step 15
    df_test_models = pd.read_csv(step15_dir / "test_metrics.csv")
    
    plot_bar_metrics_by_model(
        df_metrics=df_test_models,
        metric_col="test_mae",
        out_path=output_dir / "test_mae_by_model.png",
        title="Test MAE by Selected Models",
        ylabel="Test MAE",
    )
    logger.info("  ✓ test_mae_by_model.png")
    
    plot_bar_metrics_by_model(
        df_metrics=df_test_models,
        metric_col="test_r2",
        out_path=output_dir / "test_r2_by_model.png",
        title="Test R² by Selected Models",
        ylabel="Test R²",
    )
    logger.info("  ✓ test_r2_by_model.png")
    logger.info("")

    # -------------------------------------------------------------------------
    # D) ROLLING-WINDOW PERFORMANCE OVER TIME
    # -------------------------------------------------------------------------
    logger.info("Creating rolling-window performance plots...")
    
    # Load rolling metrics from Step 16
    df_rolling = pd.read_csv(step16_dir / "rolling_metrics_per_fold.csv")
    
    # Rolling test MAE over time (per model)
    plot_rolling_metric_over_time(
        df_rolling=df_rolling,
        metric_col="test_mae",
        out_path=output_dir / "rolling_test_mae_by_model.png",
        title="Rolling Test MAE by Model (2015–2025)",
        ylabel="Test MAE",
        models_to_plot=["baseline_mean", "ridge", "random_forest", "xgb_best", "baseline_capm"],
    )
    logger.info("  ✓ rolling_test_mae_by_model.png")

    # Rolling test R² over time (per model)
    plot_rolling_metric_over_time(
        df_rolling=df_rolling,
        metric_col="test_r2",
        out_path=output_dir / "rolling_test_r2_by_model.png",
        title="Rolling Test R² by Model (2015–2025)",
        ylabel="Test R²",
        models_to_plot=["baseline_mean", "ridge", "random_forest", "xgb_best", "baseline_capm"],
    )
    logger.info("  ✓ rolling_test_r2_by_model.png")
    logger.info("")

    # =========================================================================
    # 5) SAVE COMPLETION MARKER
    # =========================================================================
    logger.info("=" * 70)
    logger.info("SAVING COMPLETION MARKER")
    logger.info("=" * 70)
    
    step17_marker = output_dir.parent / "step_17_completed.txt"
    with step17_marker.open("w", encoding="utf-8") as f:
        f.write(
            "Step 17 completed: regression figures generated.\n"
            f"Figures directory: {output_dir}\n"
            f"\n"
            f"Generated figures:\n"
            f"  - actual_vs_pred_ridge_val.png\n"
            f"  - actual_vs_pred_ridge_test.png\n"
            f"  - actual_vs_pred_rf_test.png\n"
            f"  - residuals_hist_ridge_test.png\n"
            f"  - residuals_vs_pred_ridge_test.png\n"
            f"  - val_mae_by_model.png\n"
            f"  - val_r2_by_model.png\n"
            f"  - test_mae_by_model.png\n"
            f"  - test_r2_by_model.png\n"
            f"  - rolling_test_mae_by_model.png\n"
            f"  - rolling_test_r2_by_model.png\n"
        )
    
    logger.info(f"  ✓ Completion marker saved: {step17_marker}")
    logger.info("")

    # =========================================================================
    # 6) SUMMARY
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 17 COMPLETED: REGRESSION FIGURES SUMMARY")
    logger.info("=" * 70)
    logger.info(f"    Total figures generated: 11")
    logger.info(f"    Output directory: {output_dir}")
    logger.info("")
    logger.info("    Figure categories:")
    logger.info("      - Actual vs Predicted: 3 figures")
    logger.info("      - Residual Analysis: 2 figures")
    logger.info("      - Model Comparison: 4 figures")
    logger.info("      - Rolling Performance: 2 figures")
    logger.info("")
    logger.info("    All figures saved as PNG (200 DPI)")
    logger.info("=" * 70)

    print("\nStep 17 completed successfully: regression PNG figures saved.")


if __name__ == "__main__":
    run_step_17()
