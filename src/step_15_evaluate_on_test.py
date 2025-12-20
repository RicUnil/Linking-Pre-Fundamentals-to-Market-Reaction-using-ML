"""
Step 15 — Final Test-Set Evaluation (Ridge & Baselines)

This step:
- Loads preprocessed test data from Step 10.
- Reuses the feature imputer fitted in Step 11.
- Loads selected models trained in Steps 11–13:
    * baseline_mean
    * baseline_capm
    * ridge (final model)
    * random_forest (best tree family representative)
    * xgb_best (best boosting-style model)
- Computes predictions on the test set for each model.
- Evaluates all models on the test set using MAE, RMSE, R².
- Saves:
    * test_metrics.json
    * test_metrics.csv
    * ridge_test_residuals.parquet (id + y_true + y_pred + residual)
    * step_15_completed.txt

This step provides the final, honest out-of-sample evaluation to confront
H₀: 30-day post-earnings excess returns are unpredictable.
"""

from typing import Dict
from pathlib import Path
import json
import logging

import joblib
import numpy as np
import pandas as pd

from src.config import Settings
from src.metrics.regression import regression_metrics
from src.models.baselines import predict_capm_baseline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_selected_models(settings: Settings) -> Dict[str, object]:
    """
    Load the subset of models to be evaluated on the test set.

    Models:
    - baseline_mean        (DummyRegressor)
    - capm_betas           (DataFrame of betas)
    - ridge                (final model)
    - random_forest        (tree representative)
    - xgb_best             (best boosting-style model)

    Parameters
    ----------
    settings : Settings
        Project settings.

    Returns
    -------
    dict
        Mapping model name -> model object or betas DataFrame.
    """
    results_dir = settings.RESULTS_DIR

    models: Dict[str, object] = {}

    # Step 11 models
    step11_dir = results_dir / "step_11"
    models["baseline_mean"] = joblib.load(step11_dir / "baseline_mean_model.joblib")
    models["capm_betas"] = pd.read_parquet(step11_dir / "baseline_capm_betas.parquet")
    models["ridge"] = joblib.load(step11_dir / "ridge_model.joblib")

    # Step 12 models (tree)
    step12_dir = results_dir / "step_12"
    models["random_forest"] = joblib.load(step12_dir / "rf_model.joblib")

    # Step 13 models (xgb-style, actually HistGB fallback)
    # Based on Step 14 ranking, xgb_baseline is better than xgb_tuned
    step13_dir = results_dir / "step_13"
    models["xgb_best"] = joblib.load(step13_dir / "xgb_baseline_model.joblib")

    logger.info("  ✓ Loaded all selected models")

    return models


def run_step_15() -> None:
    """
    Execute Step 15: Final test-set evaluation.

    This function:
    1. Loads test data from Step 10
    2. Loads and applies feature imputer from Step 11
    3. Loads selected models from Steps 11-13
    4. Generates predictions on test set
    5. Computes test metrics (MAE, RMSE, R²)
    6. Saves results and residuals
    """
    logger.info("=" * 70)
    logger.info("STEP 15: FINAL TEST-SET EVALUATION (RIDGE & BASELINES)")
    logger.info("=" * 70)

    settings = Settings()
    settings.ensure_directories()

    step10_dir = settings.RESULTS_DIR / "step_10"
    step11_dir = settings.RESULTS_DIR / "step_11"
    output_dir = settings.RESULTS_DIR / "step_15"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"    Output directory: {output_dir}")
    logger.info("    " + "-" * 66)

    # 1) Load test matrices
    logger.info("Loading test data from Step 10...")
    logger.info("-" * 70)

    X_test = np.load(step10_dir / "X_test.npy")
    y_test = np.load(step10_dir / "y_test.npy")

    logger.info(f"  X_test shape: {X_test.shape}")
    logger.info(f"  y_test shape: {y_test.shape}")

    # 2) Load cleaned_test dataframe (for CAPM and residual export)
    df_test = pd.read_parquet(step10_dir / "cleaned_test.parquet")
    logger.info(f"  Loaded cleaned test dataframe: {df_test.shape}")

    # 3) Load imputer from Step 11 and apply it
    imputer_path = step11_dir / "feature_imputer.joblib"
    if not imputer_path.exists():
        raise FileNotFoundError(
            f"Feature imputer not found at {imputer_path}. "
            "Please ensure Step 11 has been run successfully."
        )

    imputer = joblib.load(imputer_path)
    X_test_imp = imputer.transform(X_test)

    logger.info(f"  ✓ Loaded and applied feature imputer")
    logger.info("    " + "-" * 66)

    # 4) Load models
    logger.info("Loading selected models...")
    logger.info("-" * 70)

    models = load_selected_models(settings)

    logger.info("    " + "-" * 66)

    # 5) Compute predictions and metrics
    logger.info("Generating test predictions and computing metrics...")
    logger.info("-" * 70)

    metrics_test: Dict[str, Dict[str, float]] = {}
    y_true = y_test

    # Store predictions for residual analysis
    predictions = {}

    # 5.1 baseline_mean
    logger.info("    Evaluating baseline_mean...")
    baseline_mean_model = models["baseline_mean"]
    y_pred_mean = baseline_mean_model.predict(np.zeros((X_test_imp.shape[0], 1)))
    predictions["baseline_mean"] = y_pred_mean
    metrics_test["baseline_mean"] = regression_metrics(y_true, y_pred_mean, prefix="test_")

    logger.info(f"  Test MAE: {metrics_test['baseline_mean']['test_mae']:.6f}")
    logger.info(f"  Test R²: {metrics_test['baseline_mean']['test_r2']:.6f}")

    # 5.2 CAPM baseline
    logger.info("    Evaluating baseline_capm...")
    capm_betas = models["capm_betas"]
    y_pred_capm = predict_capm_baseline(df_test, capm_betas, settings)
    predictions["baseline_capm"] = y_pred_capm
    metrics_test["baseline_capm"] = regression_metrics(y_true, y_pred_capm, prefix="test_")

    logger.info(f"  Test MAE: {metrics_test['baseline_capm']['test_mae']:.6f}")
    logger.info(f"  Test R²: {metrics_test['baseline_capm']['test_r2']:.6f}")

    # 5.3 ridge (final model)
    logger.info("    Evaluating ridge (final model)...")
    ridge_model = models["ridge"]
    y_pred_ridge = ridge_model.predict(X_test_imp)
    predictions["ridge"] = y_pred_ridge
    metrics_test["ridge"] = regression_metrics(y_true, y_pred_ridge, prefix="test_")

    logger.info(f"  Test MAE: {metrics_test['ridge']['test_mae']:.6f}")
    logger.info(f"  Test R²: {metrics_test['ridge']['test_r2']:.6f}")

    # 5.4 random_forest
    logger.info("    Evaluating random_forest...")
    rf_model = models["random_forest"]
    y_pred_rf = rf_model.predict(X_test_imp)
    predictions["random_forest"] = y_pred_rf
    metrics_test["random_forest"] = regression_metrics(y_true, y_pred_rf, prefix="test_")

    logger.info(f"  Test MAE: {metrics_test['random_forest']['test_mae']:.6f}")
    logger.info(f"  Test R²: {metrics_test['random_forest']['test_r2']:.6f}")

    # 5.5 xgb_best (HistGB fallback)
    logger.info("    Evaluating xgb_best (xgb_baseline)...")
    xgb_model = models["xgb_best"]
    y_pred_xgb = xgb_model.predict(X_test_imp)
    predictions["xgb_best"] = y_pred_xgb
    metrics_test["xgb_best"] = regression_metrics(y_true, y_pred_xgb, prefix="test_")

    logger.info(f"  Test MAE: {metrics_test['xgb_best']['test_mae']:.6f}")
    logger.info(f"  Test R²: {metrics_test['xgb_best']['test_r2']:.6f}")

    logger.info("    " + "-" * 66)

    # 6) Save metrics: JSON + CSV
    logger.info("Saving test metrics...")
    logger.info("-" * 70)

    metrics_json_path = output_dir / "test_metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_test, f, indent=2)

    logger.info(f"  ✓ Saved test_metrics.json")

    # Flatten to DataFrame
    rows = []
    for name, m in metrics_test.items():
        rows.append(
            {
                "model_name": name,
                "test_mae": m["test_mae"],
                "test_rmse": m["test_rmse"],
                "test_r2": m["test_r2"],
            }
        )

    df_metrics = pd.DataFrame(rows)
    # Sort by test MAE
    df_metrics = df_metrics.sort_values("test_mae").reset_index(drop=True)
    df_metrics.to_csv(output_dir / "test_metrics.csv", index=False)

    logger.info(f"  ✓ Saved test_metrics.csv")
    logger.info("    " + "-" * 66)

    # 7) Save residuals for the final model (ridge)
    logger.info("Saving Ridge residuals...")
    logger.info("-" * 70)

    ticker_col = settings.EARNINGS_TICKER_COLUMN
    date_col = settings.EARNINGS_DATE_COLUMN

    residuals_df = pd.DataFrame(
        {
            ticker_col: df_test[ticker_col].values,
            date_col: df_test[date_col].values,
            "y_true": y_true,
            "y_pred_ridge": y_pred_ridge,
            "residual_ridge": y_true - y_pred_ridge,
        }
    )

    residuals_df.to_parquet(output_dir / "ridge_test_residuals.parquet", index=False)

    logger.info(f"  ✓ Saved ridge_test_residuals.parquet ({len(residuals_df)} rows)")
    logger.info("    " + "-" * 66)

    # 8) Completion marker with summary
    logger.info("Saving completion marker...")
    logger.info("-" * 70)

    marker_path = output_dir / "step_15_completed.txt"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write("Step 15 completed: test-set evaluation for selected models.\n")
        f.write(f"Rows in test set: {y_true.shape[0]}\n")
        f.write("\n")
        f.write("Test Metrics Summary:\n")
        f.write("-" * 50 + "\n")
        for _, row in df_metrics.iterrows():
            f.write(
                f"{row['model_name']:<20} "
                f"MAE: {row['test_mae']:.6f}  "
                f"R²: {row['test_r2']:.6f}\n"
            )
        f.write("-" * 50 + "\n")
        f.write(f"\nBest model: {df_metrics.iloc[0]['model_name']}\n")
        f.write(f"Best test MAE: {df_metrics.iloc[0]['test_mae']:.6f}\n")

    logger.info(f"  ✓ Completion marker saved: {marker_path}")
    logger.info("    " + "=" * 66)

    # 9) Print summary
    logger.info("STEP 15 COMPLETED: TEST-SET EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info("    Test Metrics (sorted by MAE):")
    logger.info("-" * 70)
    logger.info(f"{'Model':<20} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<12}")
    logger.info("-" * 70)

    for _, row in df_metrics.iterrows():
        logger.info(
            f"{row['model_name']:<20} "
            f"{row['test_mae']:<12.6f} "
            f"{row['test_rmse']:<12.6f} "
            f"{row['test_r2']:<12.6f}"
        )

    logger.info("-" * 70)

    # Compare best model to baseline
    best_model = df_metrics.iloc[0]
    baseline_mean_metrics = df_metrics[df_metrics["model_name"] == "baseline_mean"].iloc[0]

    improvement = 100 * (baseline_mean_metrics["test_mae"] - best_model["test_mae"]) / baseline_mean_metrics["test_mae"]

    logger.info("    🏆 Best Model: " + best_model["model_name"])
    logger.info(f"   Test MAE: {best_model['test_mae']:.6f}")
    logger.info(f"   Test R²: {best_model['test_r2']:.6f}")
    logger.info("")
    logger.info("    📊 Performance vs Baseline:")
    logger.info(f"   Baseline (Mean) Test MAE: {baseline_mean_metrics['test_mae']:.6f}")
    logger.info(f"   Best Model Test MAE: {best_model['test_mae']:.6f}")
    logger.info(f"   Improvement: {improvement:.2f}%")
    logger.info("    " + "=" * 66)
    logger.info("Key outputs:")
    logger.info("  - test_metrics.json")
    logger.info("  - test_metrics.csv")
    logger.info("  - ridge_test_residuals.parquet")
    logger.info("")
    logger.info("    Next: Step 16+ will likely add visualizations and error analysis.")
    logger.info("=" * 70)

    print("\nStep 15 completed successfully: test metrics computed and residuals saved.")


if __name__ == "__main__":
    run_step_15()
