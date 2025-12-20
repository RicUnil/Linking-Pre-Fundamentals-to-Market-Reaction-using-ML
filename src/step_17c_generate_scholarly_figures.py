"""
Step 17c — Generate Additional Scholarly Figures and Reorganize

This script:
1. Creates additional multi-panel figures for scholarly presentation
2. Moves all figures to a centralized results/figures/ folder
3. Organizes figures by category for easy navigation
"""

from pathlib import Path
import logging
import shutil

import pandas as pd

from src.config import Settings
from src.visualization.regression_plots import (
    plot_data_overview,
    plot_model_performance_summary,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_step_17c() -> None:
    """
    Execute Step 17c: Generate additional scholarly figures and reorganize.
    """
    logger.info("=" * 70)
    logger.info("STEP 17C: ADDITIONAL SCHOLARLY FIGURES & REORGANIZATION")
    logger.info("=" * 70)
    
    settings = Settings()
    settings.ensure_directories()

    results_dir = settings.RESULTS_DIR
    step10_dir = results_dir / "step_10"
    step14_dir = results_dir / "step_14"
    step15_dir = results_dir / "step_15"
    step16_dir = results_dir / "step_16"
    
    # Old figure directory
    old_figures_dir = results_dir / "step_17" / "figures"
    
    # New centralized figures directory
    new_figures_dir = results_dir / "figures"
    new_figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"New centralized figures directory: {new_figures_dir}")
    logger.info("")

    # =========================================================================
    # 1) GENERATE NEW SCHOLARLY FIGURES
    # =========================================================================
    logger.info("=" * 70)
    logger.info("GENERATING NEW SCHOLARLY FIGURES")
    logger.info("=" * 70)
    logger.info("")

    # Load data
    logger.info("Loading data...")
    df_train = pd.read_parquet(step10_dir / "cleaned_train.parquet")
    df_val = pd.read_parquet(step10_dir / "cleaned_val.parquet")
    df_test = pd.read_parquet(step10_dir / "cleaned_test.parquet")
    
    df_val_metrics = pd.read_csv(step14_dir / "model_comparison.csv")
    df_test_metrics = pd.read_csv(step15_dir / "test_metrics.csv")
    df_rolling = pd.read_csv(step16_dir / "rolling_metrics_per_fold.csv")
    
    logger.info("  ✓ Data loaded")
    logger.info("")

    # Figure 1: Data Overview
    logger.info("Creating Figure: Data Overview (4 panels)...")
    plot_data_overview(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        out_path=old_figures_dir / "scholarly_data_overview.png",
        title="Dataset Overview and Distribution Analysis",
    )
    logger.info("  ✓ scholarly_data_overview.png")
    logger.info("")

    # Figure 2: Model Performance Summary
    logger.info("Creating Figure: Model Performance Summary (4 panels)...")
    plot_model_performance_summary(
        df_val=df_val_metrics,
        df_test=df_test_metrics,
        df_rolling=df_rolling,
        out_path=old_figures_dir / "scholarly_performance_summary.png",
        title="Model Performance Summary Across All Evaluations",
    )
    logger.info("  ✓ scholarly_performance_summary.png")
    logger.info("")

    # =========================================================================
    # 2) REORGANIZE ALL FIGURES
    # =========================================================================
    logger.info("=" * 70)
    logger.info("REORGANIZING FIGURES INTO CENTRALIZED FOLDER")
    logger.info("=" * 70)
    logger.info("")

    # Create category subdirectories
    categories = {
        'comprehensive': new_figures_dir / '01_comprehensive',
        'scholarly': new_figures_dir / '02_scholarly',
        'predictions': new_figures_dir / '03_predictions',
        'residuals': new_figures_dir / '04_residuals',
        'metrics': new_figures_dir / '05_metrics',
        'rolling': new_figures_dir / '06_rolling',
    }
    
    for cat_dir in categories.values():
        cat_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Created category subdirectories:")
    for name, path in categories.items():
        logger.info(f"  - {name}: {path.name}/")
    logger.info("")

    # Move and organize figures
    logger.info("Moving and organizing figures...")
    
    moved_count = 0
    
    # Get all PNG files from old directory
    if old_figures_dir.exists():
        for fig_file in old_figures_dir.glob("*.png"):
            filename = fig_file.name
            
            # Determine category
            if 'comprehensive' in filename:
                dest_dir = categories['comprehensive']
            elif 'scholarly' in filename:
                dest_dir = categories['scholarly']
            elif 'actual_vs_pred' in filename or 'prediction' in filename:
                dest_dir = categories['predictions']
            elif 'residual' in filename:
                dest_dir = categories['residuals']
            elif 'mae_by_model' in filename or 'r2_by_model' in filename:
                dest_dir = categories['metrics']
            elif 'rolling' in filename:
                dest_dir = categories['rolling']
            else:
                dest_dir = new_figures_dir  # Root for uncategorized
            
            # Copy file
            dest_path = dest_dir / filename
            shutil.copy2(fig_file, dest_path)
            moved_count += 1
            logger.info(f"  ✓ {filename} → {dest_dir.name}/")
    
    logger.info("")
    logger.info(f"Total figures organized: {moved_count}")
    logger.info("")

    # =========================================================================
    # 3) CREATE INDEX FILE
    # =========================================================================
    logger.info("Creating figure index...")
    
    index_path = new_figures_dir / "README.md"
    with index_path.open("w") as f:
        f.write("# Figures Directory\n\n")
        f.write("This directory contains all visualization figures for the project.\n\n")
        f.write("## Organization\n\n")
        
        for cat_name, cat_dir in categories.items():
            figures_in_cat = list(cat_dir.glob("*.png"))
            f.write(f"### {cat_dir.name}/ ({len(figures_in_cat)} figures)\n\n")
            
            for fig in sorted(figures_in_cat):
                f.write(f"- `{fig.name}`\n")
            f.write("\n")
        
        # Uncategorized
        uncategorized = list(new_figures_dir.glob("*.png"))
        if uncategorized:
            f.write(f"### Root Directory ({len(uncategorized)} figures)\n\n")
            for fig in sorted(uncategorized):
                f.write(f"- `{fig.name}`\n")
            f.write("\n")
        
        f.write("\n## Figure Descriptions\n\n")
        f.write("### Comprehensive Figures (4 panels each)\n")
        f.write("- `comprehensive_model_comparison.png`: Actual vs predicted for Ridge & RF\n")
        f.write("- `comprehensive_residual_analysis.png`: Residual distributions and patterns\n")
        f.write("- `comprehensive_metrics_comparison.png`: MAE and R² across all models\n")
        f.write("- `comprehensive_rolling_analysis.png`: Temporal performance analysis\n\n")
        
        f.write("### Scholarly Figures (4 panels each)\n")
        f.write("- `scholarly_data_overview.png`: Dataset overview and statistics\n")
        f.write("- `scholarly_performance_summary.png`: Performance summary across evaluations\n\n")
        
        f.write("### Individual Figures\n")
        f.write("- Prediction scatter plots\n")
        f.write("- Residual histograms and scatter plots\n")
        f.write("- Metric bar charts\n")
        f.write("- Rolling performance time series\n")
    
    logger.info(f"  ✓ Created README.md index")
    logger.info("")

    # =========================================================================
    # 4) SUMMARY
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 17C COMPLETED: SUMMARY")
    logger.info("=" * 70)
    logger.info(f"    New figures generated: 2")
    logger.info(f"    Total figures organized: {moved_count}")
    logger.info(f"    Centralized directory: {new_figures_dir}")
    logger.info("")
    logger.info("    Figure categories:")
    for cat_name, cat_dir in categories.items():
        count = len(list(cat_dir.glob("*.png")))
        logger.info(f"      - {cat_dir.name}/: {count} figures")
    logger.info("")
    logger.info("    All figures are now organized for easy navigation!")
    logger.info("=" * 70)

    print("\nStep 17c completed successfully: scholarly figures created and all figures reorganized.")


if __name__ == "__main__":
    run_step_17c()
