"""
Step 18 — Build Classification Labels & Datasets

This step:
- Creates a binary classification target from the continuous excess_return_30d:
    label_outperform = 1 if excess_return_30d > 0, else 0.
- Ensures the classification labels are perfectly aligned with the existing
  train/validation/test splits created in Step 10.
- Saves classification-ready label arrays:
    * y_train_class.npy
    * y_val_class.npy
    * y_test_class.npy
- Computes and saves class balance statistics (counts and proportions) per split
  and overall into class_balance_summary.json.
- Saves a small classification_dataset_spec.json documenting the setup.

This step does NOT train any classification model. It only prepares the data
for classification (Step 19 will handle model training and AUC computation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import logging

import numpy as np

from src.config import Settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_step_18() -> None:
    """
    Execute Step 18: Build classification labels from regression targets.
    
    This function:
    1. Loads continuous regression targets (y_train, y_val, y_test)
    2. Converts them to binary labels (1 = outperform, 0 = underperform)
    3. Saves classification label arrays
    4. Computes and saves class balance statistics
    5. Creates classification dataset specification
    """
    logger.info("=" * 70)
    logger.info("STEP 18: BUILD CLASSIFICATION LABELS & DATASETS")
    logger.info("=" * 70)
    
    settings = Settings()
    settings.ensure_directories()

    results_dir = settings.RESULTS_DIR
    step10_dir = results_dir / "step_10"
    output_dir = results_dir / "step_18"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # =========================================================================
    # 1) LOAD EXISTING CONTINUOUS REGRESSION TARGETS
    # =========================================================================
    logger.info("Loading continuous regression targets from Step 10...")
    
    y_train = np.load(step10_dir / "y_train.npy")
    y_val = np.load(step10_dir / "y_val.npy")
    y_test = np.load(step10_dir / "y_test.npy")
    
    logger.info(f"  Train targets: {y_train.shape}")
    logger.info(f"  Validation targets: {y_val.shape}")
    logger.info(f"  Test targets: {y_test.shape}")

    # Sanity checks
    if y_train.ndim != 1 or y_val.ndim != 1 or y_test.ndim != 1:
        raise ValueError("Expected 1D arrays for y_train/y_val/y_test.")
    
    logger.info("  ✓ Targets loaded successfully")
    logger.info("")

    # =========================================================================
    # 2) CONVERT TO BINARY LABELS
    # =========================================================================
    logger.info("Creating binary classification labels...")
    logger.info("  Rule: label_outperform = 1 if excess_return_30d > 0.0 else 0")
    logger.info("")
    
    # 1 if excess_return_30d > 0 else 0
    y_train_class = (y_train > 0.0).astype(int)
    y_val_class = (y_val > 0.0).astype(int)
    y_test_class = (y_test > 0.0).astype(int)
    
    logger.info(f"  Train labels: {y_train_class.shape}")
    logger.info(f"  Validation labels: {y_val_class.shape}")
    logger.info(f"  Test labels: {y_test_class.shape}")
    logger.info("  ✓ Binary labels created")
    logger.info("")

    # =========================================================================
    # 3) SAVE CLASSIFICATION LABELS
    # =========================================================================
    logger.info("Saving classification label arrays...")
    
    np.save(output_dir / "y_train_class.npy", y_train_class)
    np.save(output_dir / "y_val_class.npy", y_val_class)
    np.save(output_dir / "y_test_class.npy", y_test_class)
    
    logger.info("  ✓ y_train_class.npy")
    logger.info("  ✓ y_val_class.npy")
    logger.info("  ✓ y_test_class.npy")
    logger.info("")

    # =========================================================================
    # 4) COMPUTE CLASS BALANCE STATISTICS
    # =========================================================================
    logger.info("Computing class balance statistics...")
    
    def _class_stats(y: np.ndarray) -> Dict[str, Any]:
        """Compute class distribution statistics."""
        unique, counts = np.unique(y, return_counts=True)
        total = int(counts.sum())
        stats: Dict[str, Any] = {"total": total}
        for label, count in zip(unique, counts):
            p = float(count) / float(total) if total > 0 else 0.0
            stats[str(int(label))] = {
                "count": int(count),
                "proportion": p,
            }
        return stats

    train_stats = _class_stats(y_train_class)
    val_stats = _class_stats(y_val_class)
    test_stats = _class_stats(y_test_class)

    # Overall statistics across all splits
    y_all_class = np.concatenate([y_train_class, y_val_class, y_test_class], axis=0)
    overall_stats = _class_stats(y_all_class)
    
    # Log statistics
    logger.info("")
    logger.info("  Class Balance Summary:")
    logger.info("  " + "-" * 66)
    
    for split_name, stats in [("Train", train_stats), ("Validation", val_stats), 
                               ("Test", test_stats), ("Overall", overall_stats)]:
        logger.info(f"  {split_name}:")
        logger.info(f"    Total: {stats['total']}")
        if '0' in stats:
            logger.info(f"    Class 0 (underperform): {stats['0']['count']} ({stats['0']['proportion']:.2%})")
        if '1' in stats:
            logger.info(f"    Class 1 (outperform):   {stats['1']['count']} ({stats['1']['proportion']:.2%})")
        logger.info("")

    balance_summary: Dict[str, Any] = {
        "label_definition": "label_outperform = 1 if excess_return_30d > 0.0 else 0",
        "train": train_stats,
        "validation": val_stats,
        "test": test_stats,
        "overall": overall_stats,
    }

    with (output_dir / "class_balance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(balance_summary, f, indent=2)
    
    logger.info("  ✓ class_balance_summary.json saved")
    logger.info("")

    # =========================================================================
    # 5) SAVE CLASSIFICATION DATASET SPECIFICATION
    # =========================================================================
    logger.info("Creating classification dataset specification...")
    
    # Reuse feature definition from dataset_spec.json (Step 10) so that
    # classification models will use the SAME features and splits.
    dataset_spec_path = step10_dir / "dataset_spec.json"
    if dataset_spec_path.exists():
        with dataset_spec_path.open("r", encoding="utf-8") as f:
            dataset_spec_step10 = json.load(f)
        feature_columns = dataset_spec_step10.get("feature_columns", [])
    else:
        feature_columns = []

    classification_spec = {
        "task_type": "binary_classification",
        "label_name": "label_outperform",
        "label_definition": "1 if excess_return_30d > 0.0 else 0",
        "source_regression_target": "excess_return_30d",
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
        "splits": {
            "train": {
                "size": int(y_train_class.shape[0]),
                "class_0": int(train_stats.get('0', {}).get('count', 0)),
                "class_1": int(train_stats.get('1', {}).get('count', 0)),
            },
            "validation": {
                "size": int(y_val_class.shape[0]),
                "class_0": int(val_stats.get('0', {}).get('count', 0)),
                "class_1": int(val_stats.get('1', {}).get('count', 0)),
            },
            "test": {
                "size": int(y_test_class.shape[0]),
                "class_0": int(test_stats.get('0', {}).get('count', 0)),
                "class_1": int(test_stats.get('1', {}).get('count', 0)),
            },
        },
        "notes": "Uses the exact same train/validation/test splits as regression (Step 10).",
    }

    with (output_dir / "classification_dataset_spec.json").open("w", encoding="utf-8") as f:
        json.dump(classification_spec, f, indent=2)
    
    logger.info("  ✓ classification_dataset_spec.json saved")
    logger.info("")

    # =========================================================================
    # 6) SAVE COMPLETION MARKER
    # =========================================================================
    logger.info("Saving completion marker...")
    
    marker_path = output_dir / "step_18_completed.txt"
    with marker_path.open("w", encoding="utf-8") as f:
        f.write(
            "Step 18 completed: classification labels created and saved.\n"
            f"Train labels shape: {y_train_class.shape}\n"
            f"Validation labels shape: {y_val_class.shape}\n"
            f"Test labels shape: {y_test_class.shape}\n"
            f"\n"
            f"Class balance:\n"
            f"  Train: {train_stats.get('1', {}).get('count', 0)}/{train_stats['total']} outperform "
            f"({train_stats.get('1', {}).get('proportion', 0):.2%})\n"
            f"  Validation: {val_stats.get('1', {}).get('count', 0)}/{val_stats['total']} outperform "
            f"({val_stats.get('1', {}).get('proportion', 0):.2%})\n"
            f"  Test: {test_stats.get('1', {}).get('count', 0)}/{test_stats['total']} outperform "
            f"({test_stats.get('1', {}).get('proportion', 0):.2%})\n"
        )
    
    logger.info(f"  ✓ Completion marker saved: {marker_path}")
    logger.info("")

    # =========================================================================
    # 7) SUMMARY
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 18 COMPLETED: CLASSIFICATION LABELS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"    Output directory: {output_dir}")
    logger.info("")
    logger.info("    Files created:")
    logger.info("      - y_train_class.npy")
    logger.info("      - y_val_class.npy")
    logger.info("      - y_test_class.npy")
    logger.info("      - class_balance_summary.json")
    logger.info("      - classification_dataset_spec.json")
    logger.info("      - step_18_completed.txt")
    logger.info("")
    logger.info("    Label definition:")
    logger.info("      1 = outperform (excess_return_30d > 0.0)")
    logger.info("      0 = underperform or equal (excess_return_30d <= 0.0)")
    logger.info("")
    logger.info("    Overall class balance:")
    logger.info(f"      Class 0: {overall_stats.get('0', {}).get('count', 0)} "
                f"({overall_stats.get('0', {}).get('proportion', 0):.2%})")
    logger.info(f"      Class 1: {overall_stats.get('1', {}).get('count', 0)} "
                f"({overall_stats.get('1', {}).get('proportion', 0):.2%})")
    logger.info("")
    logger.info("    ✓ Classification labels ready for Step 19 (model training)")
    logger.info("=" * 70)

    print("\nStep 18 completed successfully: classification labels and summaries saved.")


if __name__ == "__main__":
    run_step_18()
