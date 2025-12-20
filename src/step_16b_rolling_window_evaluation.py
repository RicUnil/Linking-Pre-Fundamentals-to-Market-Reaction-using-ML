"""
Step 16b — Rolling-Window Out-of-Sample Evaluation

This step:
- Rebuilds a full time-ordered panel of cleaned earnings events.
- Defines time-based rolling train/test splits (e.g. yearly folds).
- For each fold:
    * Trains a small set of models on the past window:
        - baseline_mean
        - baseline_capm
        - ridge
        - random_forest
        - xgb_best
    * Evaluates on the next period (strictly forward in time).
- Computes MAE, RMSE, R² per model and per fold.
- Saves:
    * rolling_metrics_per_fold.json
    * rolling_metrics_per_fold.csv
    * rolling_metrics_aggregated.json
    * rolling_metrics_aggregated.csv
    * step_16_completed.txt

This step provides a temporal robustness check of the H₀ conclusion.
"""

from typing import Dict, Any
from pathlib import Path
import json
import logging

import pandas as pd
import numpy as np

from src.config import Settings
from src.evaluation.rolling_window import run_rolling_window_evaluation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_step_16b() -> None:
    """
    Execute Step 16: Rolling-window out-of-sample evaluation.

    This function:
    1. Rebuilds full time-ordered panel from Step 10
    2. Generates time-based rolling splits
    3. Trains models on each training window
    4. Evaluates on each test window
    5. Saves per-fold and aggregated metrics
    """
    logger.info("=" * 70)
    logger.info("STEP 16: ROLLING-WINDOW OUT-OF-SAMPLE EVALUATION")
    logger.info("=" * 70)
    
    settings = Settings()
    settings.ensure_directories()

    output_dir = settings.RESULTS_DIR / "step_16b"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"    Output directory: {output_dir}")
    logger.info("    " + "-" * 66)
    logger.info("")

    # Run rolling-window evaluation
    results = run_rolling_window_evaluation(settings, min_train_years=5)

    logger.info("")
    logger.info("=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)

    # Serialize per-fold metrics
    rows = []
    for res in results:
        for model_name, m in res.metrics.items():
            row: Dict[str, Any] = {
                "fold_id": res.fold_id,
                "train_start": res.train_start,
                "train_end": res.train_end,
                "test_start": res.test_start,
                "test_end": res.test_end,
                "model_name": model_name,
            }
            row.update(m)
            rows.append(row)

    df_metrics = pd.DataFrame(rows)

    # Save CSV + JSON (per-fold)
    csv_path = output_dir / "rolling_metrics_per_fold.csv"
    json_path = output_dir / "rolling_metrics_per_fold.json"
    
    df_metrics.to_csv(csv_path, index=False)
    df_metrics.to_json(json_path, orient="records", indent=2)
    
    logger.info(f"  ✓ Saved rolling_metrics_per_fold.csv ({len(df_metrics)} rows)")
    logger.info(f"  ✓ Saved rolling_metrics_per_fold.json")

    # Aggregate metrics per model (mean over folds)
    agg = (
        df_metrics.groupby("model_name")[["test_mae", "test_rmse", "test_r2"]]
        .agg(['mean', 'std', 'min', 'max'])
        .reset_index()
    )
    
    # Flatten column names
    agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in agg.columns.values]
    
    agg_csv_path = output_dir / "rolling_metrics_aggregated.csv"
    agg_json_path = output_dir / "rolling_metrics_aggregated.json"
    
    agg.to_csv(agg_csv_path, index=False)
    agg.to_json(agg_json_path, orient="records", indent=2)
    
    logger.info(f"  ✓ Saved rolling_metrics_aggregated.csv")
    logger.info(f"  ✓ Saved rolling_metrics_aggregated.json")

    # Completion marker
    marker_path = output_dir / "step_16b_completed.txt"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write(
            "Step 16b completed: rolling-window OOS evaluation across time folds.\n"
            f"Number of folds: {len(results)}\n"
            f"\n"
        )
        
        # Add summary statistics
        f.write("Aggregated Test Metrics (mean across folds):\n")
        f.write("-" * 50 + "\n")
        
        # Compute simple mean per model
        agg_simple = df_metrics.groupby("model_name")[["test_mae", "test_rmse", "test_r2"]].mean()
        
        for model_name in agg_simple.index:
            mae = agg_simple.loc[model_name, "test_mae"]
            r2 = agg_simple.loc[model_name, "test_r2"]
            f.write(f"{model_name:<20} MAE: {mae:.6f}  R²: {r2:.6f}\n")
        
        f.write("-" * 50 + "\n")
        
        # Best model
        best_model = agg_simple["test_mae"].idxmin()
        best_mae = agg_simple.loc[best_model, "test_mae"]
        f.write(f"\nBest model (avg): {best_model}\n")
        f.write(f"Best avg test MAE: {best_mae:.6f}\n")

    logger.info(f"  ✓ Completion marker saved: {marker_path}")
    logger.info("")

    # Print summary
    logger.info("=" * 70)
    logger.info("STEP 16b COMPLETED: ROLLING-WINDOW EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"    Number of folds: {len(results)}")
    logger.info("")
    logger.info("    Aggregated Test Metrics (mean ± std across folds):")
    logger.info("-" * 70)
    logger.info(f"{'Model':<20} {'Mean MAE':<15} {'Std MAE':<15} {'Mean R²':<15}")
    logger.info("-" * 70)
    
    # Compute aggregated stats
    agg_stats = df_metrics.groupby("model_name")[["test_mae", "test_r2"]].agg(['mean', 'std'])
    
    for model_name in agg_stats.index:
        mean_mae = agg_stats.loc[model_name, ('test_mae', 'mean')]
        std_mae = agg_stats.loc[model_name, ('test_mae', 'std')]
        mean_r2 = agg_stats.loc[model_name, ('test_r2', 'mean')]
        
        logger.info(
            f"{model_name:<20} "
            f"{mean_mae:<15.6f} "
            f"{std_mae:<15.6f} "
            f"{mean_r2:<15.6f}"
        )
    
    logger.info("-" * 70)
    
    # Best model
    best_model = agg_stats[('test_mae', 'mean')].idxmin()
    best_mae = agg_stats.loc[best_model, ('test_mae', 'mean')]
    baseline_mae = agg_stats.loc['baseline_mean', ('test_mae', 'mean')]
    improvement = 100 * (baseline_mae - best_mae) / baseline_mae
    
    logger.info("")
    logger.info(f"    🏆 Best Model (avg across folds): {best_model}")
    logger.info(f"   Avg Test MAE: {best_mae:.6f}")
    logger.info("")
    logger.info(f"    📊 Performance vs Baseline:")
    logger.info(f"   Baseline (Mean) Avg Test MAE: {baseline_mae:.6f}")
    logger.info(f"   Best Model Avg Test MAE: {best_mae:.6f}")
    logger.info(f"   Improvement: {improvement:.2f}%")
    logger.info("")
    logger.info("    " + "=" * 66)
    logger.info("Key outputs:")
    logger.info("  - rolling_metrics_per_fold.csv")
    logger.info("  - rolling_metrics_per_fold.json")
    logger.info("  - rolling_metrics_aggregated.csv")
    logger.info("  - rolling_metrics_aggregated.json")
    logger.info("")
    logger.info("    Next: Step 17+ will likely add visualizations of rolling performance.")
    logger.info("=" * 70)

    print("\nStep 16b completed successfully: rolling-window OOS metrics saved.")


if __name__ == "__main__":
    run_step_16b()
