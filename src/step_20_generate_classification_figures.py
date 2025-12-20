"""
Step 20 — Classification Visualizations (ROC, Confusion Matrix, AUC Barplots)

This step:
- Loads classification models (Step 19), classification labels (Step 18),
  and feature matrices (Step 10).
- Applies the same feature imputer as in regression/classification training.
- Regenerates predictions and probabilities on validation and test sets.
- Creates PNG figures:
    * ROC curves (val & test, multiple models)
    * Confusion matrices for the best model (val & test)
    * Barplots of ROC-AUC by model (val & test)
- Saves figures under results/step_20/figures_classification/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import logging

import joblib
import numpy as np
import pandas as pd

from src.config import Settings
from src.visualization.classification_plots import (
    plot_roc_curves_multi_model,
    plot_confusion_matrix,
    plot_bar_auc_by_model,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_step_20() -> None:
    """
    Execute Step 20: Generate classification visualization figures.
    
    This function:
    1. Loads feature matrices, classification labels, and trained models
    2. Applies feature imputation
    3. Recomputes predictions and probabilities
    4. Generates comprehensive classification visualizations:
       - ROC curves for all models (validation & test)
       - Confusion matrices for best model (validation & test)
       - AUC bar plots (validation & test)
    5. Saves all figures as PNG files
    """
    logger.info("=" * 70)
    logger.info("STEP 20: CLASSIFICATION VISUALIZATIONS")
    logger.info("=" * 70)
    logger.info("")
    
    settings = Settings()
    settings.ensure_directories()

    results_dir = settings.RESULTS_DIR
    step10_dir = results_dir / "step_10"
    step11_dir = results_dir / "step_11"
    step18_dir = results_dir / "step_18"
    step19_dir = results_dir / "step_19"

    output_dir = results_dir / "step_20" / "figures_classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # =========================================================================
    # 1) LOAD FEATURE MATRICES AND LABELS
    # =========================================================================
    logger.info("Loading feature matrices and classification labels...")
    
    X_val = np.load(step10_dir / "X_val.npy")
    X_test = np.load(step10_dir / "X_test.npy")
    y_val = np.load(step18_dir / "y_val_class.npy")
    y_test = np.load(step18_dir / "y_test_class.npy")
    
    logger.info(f"  Validation features: {X_val.shape}")
    logger.info(f"  Test features: {X_test.shape}")
    logger.info(f"  Validation labels: {y_val.shape}")
    logger.info(f"  Test labels: {y_test.shape}")
    logger.info("  ✓ Data loaded")
    logger.info("")

    # =========================================================================
    # 2) APPLY FEATURE IMPUTER
    # =========================================================================
    logger.info("Applying feature imputer from Step 11...")
    
    imputer = joblib.load(step11_dir / "feature_imputer.joblib")
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    
    logger.info(f"  Imputer: {type(imputer).__name__}")
    logger.info("  ✓ Features imputed")
    logger.info("")

    # =========================================================================
    # 3) LOAD TRAINED CLASSIFICATION MODELS
    # =========================================================================
    logger.info("Loading trained classification models from Step 19...")
    
    models: Dict[str, object] = {
        "dummy_most_frequent": joblib.load(step19_dir / "dummy_most_frequent.joblib"),
        "logistic_regression": joblib.load(step19_dir / "logistic_regression.joblib"),
        "random_forest_classifier": joblib.load(step19_dir / "random_forest_classifier.joblib"),
        "gradient_boosting_classifier": joblib.load(step19_dir / "gradient_boosting_classifier.joblib"),
    }
    
    logger.info(f"  ✓ {len(models)} models loaded")
    logger.info("")

    # =========================================================================
    # 4) RECOMPUTE PREDICTIONS & PROBABILITIES
    # =========================================================================
    logger.info("Recomputing predictions and probabilities...")
    
    proba_val: Dict[str, np.ndarray] = {}
    proba_test: Dict[str, np.ndarray] = {}
    pred_val: Dict[str, np.ndarray] = {}
    pred_test: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        # Class predictions
        y_val_pred = model.predict(X_val_imp)
        y_test_pred = model.predict(X_test_imp)
        pred_val[name] = y_val_pred
        pred_test[name] = y_test_pred

        # Positive-class probability or score
        if hasattr(model, "predict_proba"):
            proba_val[name] = model.predict_proba(X_val_imp)[:, 1]
            proba_test[name] = model.predict_proba(X_test_imp)[:, 1]
        elif hasattr(model, "decision_function"):
            proba_val[name] = model.decision_function(X_val_imp)
            proba_test[name] = model.decision_function(X_test_imp)
        else:
            proba_val[name] = None
            proba_test[name] = None
    
    logger.info("  ✓ Predictions and probabilities computed")
    logger.info("")

    # =========================================================================
    # 5) LOAD CLASSIFICATION METRICS
    # =========================================================================
    logger.info("Loading classification metrics from Step 19...")
    
    df_metrics = pd.read_csv(step19_dir / "classification_metrics_summary.csv")
    
    logger.info(f"  ✓ Metrics loaded ({len(df_metrics)} rows)")
    logger.info("")

    # =========================================================================
    # 6) GENERATE VISUALIZATIONS
    # =========================================================================
    logger.info("=" * 70)
    logger.info("GENERATING CLASSIFICATION FIGURES")
    logger.info("=" * 70)
    logger.info("")
    
    # 6.1) AUC Bar Plots
    logger.info("1. Generating AUC bar plots...")
    
    plot_bar_auc_by_model(
        df_metrics=df_metrics,
        split="val",
        out_path=output_dir / "auc_by_model_val.png",
        title="Validation ROC-AUC by Model",
    )
    logger.info("  ✓ auc_by_model_val.png")

    plot_bar_auc_by_model(
        df_metrics=df_metrics,
        split="test",
        out_path=output_dir / "auc_by_model_test.png",
        title="Test ROC-AUC by Model",
    )
    logger.info("  ✓ auc_by_model_test.png")
    logger.info("")

    # 6.2) ROC Curves
    logger.info("2. Generating ROC curves...")
    
    plot_roc_curves_multi_model(
        y_true=y_val,
        proba_by_model=proba_val,
        out_path=output_dir / "roc_curves_val.png",
        title="ROC Curves (Validation Set)",
    )
    logger.info("  ✓ roc_curves_val.png")

    plot_roc_curves_multi_model(
        y_true=y_test,
        proba_by_model=proba_test,
        out_path=output_dir / "roc_curves_test.png",
        title="ROC Curves (Test Set)",
    )
    logger.info("  ✓ roc_curves_test.png")
    logger.info("")

    # 6.3) Confusion Matrices for Best Model
    logger.info("3. Generating confusion matrices for best model...")
    
    # Identify best model by validation AUC
    df_val = df_metrics[df_metrics["split"] == "val"].copy()
    if not df_val.empty and "roc_auc" in df_val.columns:
        best_row = df_val.sort_values("roc_auc", ascending=False).iloc[0]
        best_model_name = str(best_row["model_name"])
        best_val_auc = float(best_row["roc_auc"])
    else:
        best_model_name = "gradient_boosting_classifier"  # fallback
        best_val_auc = 0.0
    
    logger.info(f"  Best model (by validation AUC): {best_model_name} (AUC = {best_val_auc:.4f})")
    logger.info("")

    y_val_pred_best = pred_val[best_model_name]
    y_test_pred_best = pred_test[best_model_name]

    labels_display = ["Underperform (0)", "Outperform (1)"]

    # Raw confusion matrices
    plot_confusion_matrix(
        y_true=y_val,
        y_pred=y_val_pred_best,
        out_path=output_dir / f"confusion_matrix_{best_model_name}_val.png",
        title=f"Confusion Matrix (Validation) — {best_model_name}",
        normalize=False,
        labels=labels_display,
    )
    logger.info(f"  ✓ confusion_matrix_{best_model_name}_val.png")

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred_best,
        out_path=output_dir / f"confusion_matrix_{best_model_name}_test.png",
        title=f"Confusion Matrix (Test) — {best_model_name}",
        normalize=False,
        labels=labels_display,
    )
    logger.info(f"  ✓ confusion_matrix_{best_model_name}_test.png")

    # Normalized confusion matrices
    plot_confusion_matrix(
        y_true=y_val,
        y_pred=y_val_pred_best,
        out_path=output_dir / f"confusion_matrix_{best_model_name}_val_normalized.png",
        title=f"Normalized Confusion Matrix (Validation) — {best_model_name}",
        normalize=True,
        labels=labels_display,
    )
    logger.info(f"  ✓ confusion_matrix_{best_model_name}_val_normalized.png")

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred_best,
        out_path=output_dir / f"confusion_matrix_{best_model_name}_test_normalized.png",
        title=f"Normalized Confusion Matrix (Test) — {best_model_name}",
        normalize=True,
        labels=labels_display,
    )
    logger.info(f"  ✓ confusion_matrix_{best_model_name}_test_normalized.png")
    logger.info("")

    # =========================================================================
    # 7) SAVE COMPLETION MARKER
    # =========================================================================
    logger.info("Saving completion marker...")
    
    marker_path = output_dir.parent / "step_20_completed.txt"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write("Step 20 completed: classification figures generated.\n")
        f.write(f"Figures directory: {output_dir}\n")
        f.write(f"Best validation model (by ROC-AUC): {best_model_name}\n")
        f.write(f"Best validation AUC: {best_val_auc:.4f}\n")
        f.write("\n")
        f.write("Figures created:\n")
        f.write("  - roc_curves_val.png\n")
        f.write("  - roc_curves_test.png\n")
        f.write("  - auc_by_model_val.png\n")
        f.write("  - auc_by_model_test.png\n")
        f.write(f"  - confusion_matrix_{best_model_name}_val.png\n")
        f.write(f"  - confusion_matrix_{best_model_name}_test.png\n")
        f.write(f"  - confusion_matrix_{best_model_name}_val_normalized.png\n")
        f.write(f"  - confusion_matrix_{best_model_name}_test_normalized.png\n")
    
    logger.info(f"  ✓ Completion marker saved: {marker_path}")
    logger.info("")

    # =========================================================================
    # 8) FINAL SUMMARY
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 20 COMPLETED: CLASSIFICATION FIGURES SUMMARY")
    logger.info("=" * 70)
    logger.info(f"    Output directory: {output_dir}")
    logger.info("")
    logger.info("    Figures created:")
    logger.info("      ROC Curves:")
    logger.info("        - roc_curves_val.png")
    logger.info("        - roc_curves_test.png")
    logger.info("")
    logger.info("      AUC Bar Plots:")
    logger.info("        - auc_by_model_val.png")
    logger.info("        - auc_by_model_test.png")
    logger.info("")
    logger.info("      Confusion Matrices:")
    logger.info(f"        - confusion_matrix_{best_model_name}_val.png")
    logger.info(f"        - confusion_matrix_{best_model_name}_test.png")
    logger.info(f"        - confusion_matrix_{best_model_name}_val_normalized.png")
    logger.info(f"        - confusion_matrix_{best_model_name}_test_normalized.png")
    logger.info("")
    logger.info(f"    Best model: {best_model_name} (Validation AUC = {best_val_auc:.4f})")
    logger.info("")
    logger.info("    ✓ All classification visualizations complete!")
    logger.info("    ✓ Ready for report and presentation")
    logger.info("=" * 70)

    print("\nStep 20 completed successfully: classification PNG figures saved.")


if __name__ == "__main__":
    run_step_20()
