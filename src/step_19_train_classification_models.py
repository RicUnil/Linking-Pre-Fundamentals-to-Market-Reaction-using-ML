"""
Step 19 — Train Classification Models & Compute AUC

This step:
- Loads classification labels from Step 18 and features from Step 10.
- Applies the same feature imputer used in regression (Step 11).
- Trains several classification models:
    * dummy_most_frequent (baseline)
    * logistic_regression
    * random_forest_classifier
    * gradient_boosting_classifier
- Evaluates each model on train, validation, and test sets using:
    * Accuracy
    * Balanced accuracy
    * Precision
    * Recall
    * F1-score
    * ROC-AUC (main metric)
- Saves:
    * Trained models (.joblib)
    * Predictions/probabilities for validation and test sets
    * classification_metrics_summary.json
    * classification_metrics_summary.csv
    * step_19_completed.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import logging

import joblib
import numpy as np
import pandas as pd

from src.config import Settings
from src.models.classification_models import train_classification_models


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_step_19() -> None:
    """
    Execute Step 19: Train classification models and compute AUC scores.
    
    This function:
    1. Loads feature matrices and classification labels
    2. Applies feature imputation
    3. Trains 4 classification models
    4. Evaluates on train/val/test with ROC-AUC and other metrics
    5. Saves models and comprehensive metrics
    """
    logger.info("=" * 70)
    logger.info("STEP 19: TRAIN CLASSIFICATION MODELS & COMPUTE AUC")
    logger.info("=" * 70)
    
    settings = Settings()
    settings.ensure_directories()

    results_dir = settings.RESULTS_DIR
    step10_dir = results_dir / "step_10"
    step11_dir = results_dir / "step_11"
    step18_dir = results_dir / "step_18"
    output_dir = results_dir / "step_19"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # =========================================================================
    # 1) LOAD FEATURE MATRICES (ALREADY SCALED) FROM STEP 10
    # =========================================================================
    logger.info("Loading feature matrices from Step 10...")
    
    X_train = np.load(step10_dir / "X_train.npy")
    X_val = np.load(step10_dir / "X_val.npy")
    X_test = np.load(step10_dir / "X_test.npy")
    
    logger.info(f"  Train features: {X_train.shape}")
    logger.info(f"  Validation features: {X_val.shape}")
    logger.info(f"  Test features: {X_test.shape}")
    logger.info("  ✓ Features loaded")
    logger.info("")

    # =========================================================================
    # 2) LOAD CLASSIFICATION LABELS FROM STEP 18
    # =========================================================================
    logger.info("Loading classification labels from Step 18...")
    
    y_train_class = np.load(step18_dir / "y_train_class.npy")
    y_val_class = np.load(step18_dir / "y_val_class.npy")
    y_test_class = np.load(step18_dir / "y_test_class.npy")
    
    logger.info(f"  Train labels: {y_train_class.shape}")
    logger.info(f"  Validation labels: {y_val_class.shape}")
    logger.info(f"  Test labels: {y_test_class.shape}")

    # Sanity checks
    assert X_train.shape[0] == y_train_class.shape[0], "Train size mismatch"
    assert X_val.shape[0] == y_val_class.shape[0], "Val size mismatch"
    assert X_test.shape[0] == y_test_class.shape[0], "Test size mismatch"
    
    logger.info("  ✓ Labels loaded and aligned with features")
    logger.info("")

    # =========================================================================
    # 3) APPLY FEATURE IMPUTER (SAME AS REGRESSION)
    # =========================================================================
    logger.info("Applying feature imputer from Step 11...")
    
    imputer_path = step11_dir / "feature_imputer.joblib"
    imputer = joblib.load(imputer_path)

    X_train_imp = imputer.transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    
    logger.info(f"  Imputer: {type(imputer).__name__}")
    logger.info("  ✓ Features imputed")
    logger.info("")

    # =========================================================================
    # 4) TRAIN CLASSIFICATION MODELS AND COMPUTE METRICS
    # =========================================================================
    logger.info("=" * 70)
    logger.info("TRAINING CLASSIFICATION MODELS")
    logger.info("=" * 70)
    logger.info("")
    
    logger.info("Models to train:")
    logger.info("  1. dummy_most_frequent (baseline)")
    logger.info("  2. logistic_regression")
    logger.info("  3. random_forest_classifier")
    logger.info("  4. gradient_boosting_classifier")
    logger.info("")
    
    logger.info("Training models...")
    
    models, metrics = train_classification_models(
        X_train=X_train_imp,
        y_train=y_train_class,
        X_val=X_val_imp,
        y_val=y_val_class,
        X_test=X_test_imp,
        y_test=y_test_class,
    )
    
    logger.info(f"  ✓ {len(models)} models trained successfully")
    logger.info("")

    # =========================================================================
    # 5) SAVE TRAINED MODELS
    # =========================================================================
    logger.info("Saving trained models...")
    
    for name, model in models.items():
        model_path = output_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"  ✓ {name}.joblib")
    
    logger.info("")

    # =========================================================================
    # 6) SAVE METRICS TO JSON AND CSV
    # =========================================================================
    logger.info("Saving classification metrics...")
    
    # Save JSON
    metrics_path_json = output_dir / "classification_metrics_summary.json"
    with metrics_path_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("  ✓ classification_metrics_summary.json")

    # Flatten metrics for CSV
    rows = []
    for model_name, splits in metrics.items():
        for split_name, m in splits.items():
            row: Dict[str, Any] = {
                "model_name": model_name,
                "split": split_name,
            }
            row.update(m)
            rows.append(row)

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(output_dir / "classification_metrics_summary.csv", index=False)
    logger.info("  ✓ classification_metrics_summary.csv")
    logger.info("")

    # =========================================================================
    # 7) DISPLAY RESULTS SUMMARY
    # =========================================================================
    logger.info("=" * 70)
    logger.info("CLASSIFICATION RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("")
    
    # Display test AUC scores (main metric)
    logger.info("TEST SET ROC-AUC SCORES (Main Metric):")
    logger.info("-" * 70)
    
    test_aucs = []
    for model_name in models.keys():
        test_auc = metrics[model_name]["test"]["roc_auc"]
        test_aucs.append((model_name, test_auc))
        logger.info(f"  {model_name:35s} AUC = {test_auc:.4f}")
    
    logger.info("")
    
    # Display validation AUC scores
    logger.info("VALIDATION SET ROC-AUC SCORES:")
    logger.info("-" * 70)
    
    for model_name in models.keys():
        val_auc = metrics[model_name]["val"]["roc_auc"]
        logger.info(f"  {model_name:35s} AUC = {val_auc:.4f}")
    
    logger.info("")
    
    # Display all metrics for best model on test set
    best_model_name = max(test_aucs, key=lambda x: x[1])[0]
    logger.info(f"BEST MODEL ON TEST SET: {best_model_name}")
    logger.info("-" * 70)
    
    best_metrics = metrics[best_model_name]["test"]
    logger.info(f"  Accuracy:          {best_metrics['accuracy']:.4f}")
    logger.info(f"  Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Precision:         {best_metrics['precision']:.4f}")
    logger.info(f"  Recall:            {best_metrics['recall']:.4f}")
    logger.info(f"  F1-Score:          {best_metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:           {best_metrics['roc_auc']:.4f}")
    logger.info("")

    # =========================================================================
    # 8) SAVE COMPLETION MARKER
    # =========================================================================
    logger.info("Saving completion marker...")
    
    marker_path = output_dir / "step_19_completed.txt"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write("Step 19 completed: classification models trained and evaluated.\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write("\n")
        f.write("Models trained:\n")
        for name in models.keys():
            f.write(f"  - {name}\n")
        f.write("\n")
        f.write("Test Set ROC-AUC Scores:\n")
        for model_name, test_auc in sorted(test_aucs, key=lambda x: -x[1]):
            f.write(f"  {model_name}: {test_auc:.4f}\n")
        f.write("\n")
        f.write(f"Best model: {best_model_name} (AUC = {metrics[best_model_name]['test']['roc_auc']:.4f})\n")
    
    logger.info(f"  ✓ Completion marker saved: {marker_path}")
    logger.info("")

    # =========================================================================
    # 9) FINAL SUMMARY
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 19 COMPLETED: CLASSIFICATION MODELS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"    Output directory: {output_dir}")
    logger.info("")
    logger.info("    Files created:")
    logger.info("      - dummy_most_frequent.joblib")
    logger.info("      - logistic_regression.joblib")
    logger.info("      - random_forest_classifier.joblib")
    logger.info("      - gradient_boosting_classifier.joblib")
    logger.info("      - classification_metrics_summary.json")
    logger.info("      - classification_metrics_summary.csv")
    logger.info("      - step_19_completed.txt")
    logger.info("")
    logger.info("    Key Findings:")
    logger.info(f"      Best Test AUC: {max(test_aucs, key=lambda x: x[1])[1]:.4f} ({best_model_name})")
    logger.info(f"      Baseline AUC:  {metrics['dummy_most_frequent']['test']['roc_auc']:.4f} (dummy_most_frequent)")
    logger.info("")
    
    # Interpretation
    best_auc = max(test_aucs, key=lambda x: x[1])[1]
    if best_auc < 0.55:
        logger.info("    Interpretation:")
        logger.info("      All models achieve AUC ≈ 0.50 (random guessing).")
        logger.info("      This confirms H₀: 30-day excess returns are unpredictable,")
        logger.info("      even in a binary classification setting (outperform vs underperform).")
    elif best_auc < 0.60:
        logger.info("    Interpretation:")
        logger.info("      Models show minimal predictive power (AUC slightly > 0.50).")
        logger.info("      This provides weak evidence against H₀, but performance is")
        logger.info("      too low for practical use.")
    else:
        logger.info("    Interpretation:")
        logger.info("      Models show some predictive power (AUC > 0.60).")
        logger.info("      This suggests weak evidence against H₀.")
    
    logger.info("")
    logger.info("    ✓ Classification analysis complete!")
    logger.info("    ✓ AUC scores available for all models")
    logger.info("=" * 70)

    print("\nStep 19 completed successfully: classification models and metrics saved.")


if __name__ == "__main__":
    run_step_19()
