"""
Step 17b — Generate Comprehensive Multi-Panel Figures

This script creates additional 4-panel figures that provide more explicit
and comprehensive visualization of the regression analysis results.

Each figure contains 4 related panels to tell a complete story:
1. Model Comparison (4 panels): Predictions on validation and test sets
2. Residual Analysis (4 panels): Residual distributions and patterns
3. Metrics Comparison (4 panels): MAE and R² across validation and test
4. Rolling Analysis (4 panels): Temporal performance and variability
"""

from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd

from src.config import Settings
from src.visualization.regression_plots import (
    plot_comprehensive_model_comparison,
    plot_comprehensive_residual_analysis,
    plot_comprehensive_metrics_comparison,
    plot_comprehensive_rolling_analysis,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_step_17b() -> None:
    """
    Execute Step 17b: Generate comprehensive multi-panel figures.

    This function creates 4 comprehensive figures, each with 4 panels:
    1. Model comparison across validation and test sets
    2. Residual analysis for Ridge and Random Forest
    3. Metrics comparison (MAE and R²) across all models
    4. Rolling-window analysis with temporal patterns
    """
    logger.info("=" * 70)
    logger.info("STEP 17B: COMPREHENSIVE MULTI-PANEL FIGURES")
    logger.info("=" * 70)
    
    settings = Settings()
    settings.ensure_directories()

    results_dir = settings.RESULTS_DIR
    step10_dir = results_dir / "step_10"
    step11_dir = results_dir / "step_11"
    step12_dir = results_dir / "step_12"
    step14_dir = results_dir / "step_14"
    step15_dir = results_dir / "step_15"
    step16_dir = results_dir / "step_16"

    output_dir = results_dir / "step_17" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # =========================================================================
    # 1) LOAD DATA AND MODELS
    # =========================================================================
    logger.info("Loading data and models...")
    
    X_val = np.load(step10_dir / "X_val.npy")
    y_val = np.load(step10_dir / "y_val.npy")
    X_test = np.load(step10_dir / "X_test.npy")
    y_test = np.load(step10_dir / "y_test.npy")
    
    imputer = joblib.load(step11_dir / "feature_imputer.joblib")
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    
    ridge_model = joblib.load(step11_dir / "ridge_model.joblib")
    rf_model = joblib.load(step12_dir / "rf_model.joblib")
    
    logger.info("  ✓ Data and models loaded")
    logger.info("")

    # =========================================================================
    # 2) GENERATE PREDICTIONS
    # =========================================================================
    logger.info("Generating predictions...")
    
    predictions_dict = {
        'ridge_val': ridge_model.predict(X_val_imp),
        'ridge_test': ridge_model.predict(X_test_imp),
        'rf_val': rf_model.predict(X_val_imp),
        'rf_test': rf_model.predict(X_test_imp),
    }
    
    logger.info("  ✓ Predictions generated")
    logger.info("")

    # =========================================================================
    # 3) LOAD METRICS
    # =========================================================================
    logger.info("Loading metrics from previous steps...")
    
    df_val = pd.read_csv(step14_dir / "model_comparison.csv")
    df_test = pd.read_csv(step15_dir / "test_metrics.csv")
    df_rolling = pd.read_csv(step16_dir / "rolling_metrics_per_fold.csv")
    
    logger.info("  ✓ Metrics loaded")
    logger.info("")

    # =========================================================================
    # 4) GENERATE COMPREHENSIVE FIGURES
    # =========================================================================
    logger.info("=" * 70)
    logger.info("GENERATING COMPREHENSIVE MULTI-PANEL FIGURES")
    logger.info("=" * 70)
    logger.info("")

    # -------------------------------------------------------------------------
    # FIGURE 1: MODEL COMPARISON (4 panels)
    # -------------------------------------------------------------------------
    logger.info("Creating Figure 1: Model Comparison (4 panels)...")
    
    plot_comprehensive_model_comparison(
        y_val=y_val,
        y_test=y_test,
        predictions_dict=predictions_dict,
        out_path=output_dir / "comprehensive_model_comparison.png",
        title="Model Comparison: Actual vs Predicted (Ridge & Random Forest)",
    )
    
    logger.info("  ✓ comprehensive_model_comparison.png")
    logger.info("    Panels: Ridge Val, Ridge Test, RF Val, RF Test")
    logger.info("")

    # -------------------------------------------------------------------------
    # FIGURE 2: RESIDUAL ANALYSIS (4 panels)
    # -------------------------------------------------------------------------
    logger.info("Creating Figure 2: Residual Analysis (4 panels)...")
    
    plot_comprehensive_residual_analysis(
        y_test=y_test,
        predictions_dict=predictions_dict,
        out_path=output_dir / "comprehensive_residual_analysis.png",
        title="Residual Analysis: Distribution and Patterns (Ridge & Random Forest)",
    )
    
    logger.info("  ✓ comprehensive_residual_analysis.png")
    logger.info("    Panels: Ridge Hist, Ridge vs Pred, RF Hist, RF vs Pred")
    logger.info("")

    # -------------------------------------------------------------------------
    # FIGURE 3: METRICS COMPARISON (4 panels)
    # -------------------------------------------------------------------------
    logger.info("Creating Figure 3: Metrics Comparison (4 panels)...")
    
    plot_comprehensive_metrics_comparison(
        df_val=df_val,
        df_test=df_test,
        out_path=output_dir / "comprehensive_metrics_comparison.png",
        title="Metrics Comparison: MAE and R² Across All Models",
    )
    
    logger.info("  ✓ comprehensive_metrics_comparison.png")
    logger.info("    Panels: Val MAE, Test MAE, Val R², Test R²")
    logger.info("")

    # -------------------------------------------------------------------------
    # FIGURE 4: ROLLING ANALYSIS (4 panels)
    # -------------------------------------------------------------------------
    logger.info("Creating Figure 4: Rolling-Window Analysis (4 panels)...")
    
    plot_comprehensive_rolling_analysis(
        df_rolling=df_rolling,
        out_path=output_dir / "comprehensive_rolling_analysis.png",
        title="Rolling-Window Analysis: Temporal Performance (2015-2025)",
    )
    
    logger.info("  ✓ comprehensive_rolling_analysis.png")
    logger.info("    Panels: MAE over time, R² over time, MAE variability, Avg metrics")
    logger.info("")

    # =========================================================================
    # 5) SUMMARY
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 17B COMPLETED: COMPREHENSIVE FIGURES SUMMARY")
    logger.info("=" * 70)
    logger.info(f"    Total comprehensive figures: 4")
    logger.info(f"    Total panels: 16 (4 panels per figure)")
    logger.info(f"    Output directory: {output_dir}")
    logger.info("")
    logger.info("    Generated figures:")
    logger.info("      1. comprehensive_model_comparison.png")
    logger.info("         → Actual vs Predicted for Ridge & RF (Val & Test)")
    logger.info("      2. comprehensive_residual_analysis.png")
    logger.info("         → Residual distributions and patterns")
    logger.info("      3. comprehensive_metrics_comparison.png")
    logger.info("         → MAE and R² comparison across all models")
    logger.info("      4. comprehensive_rolling_analysis.png")
    logger.info("         → Temporal performance and variability")
    logger.info("")
    logger.info("    All figures saved at 200 DPI (publication quality)")
    logger.info("=" * 70)

    print("\nStep 17b completed successfully: comprehensive multi-panel figures saved.")


if __name__ == "__main__":
    run_step_17b()
