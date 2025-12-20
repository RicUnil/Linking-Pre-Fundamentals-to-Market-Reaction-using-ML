"""
Main Entry Point for Earnings Post-Announcement Excess Return Prediction.

This script runs the complete 22-step ML pipeline for predicting post-earnings
excess returns using machine learning models.

Research Question: Can we predict 30-day excess returns after earnings announcements?

Pipeline Overview:
    1. Data Loading (Steps 1-7): Load earnings, prices, and fundamentals
    2. Feature Engineering (Steps 8-10): Build fundamental and market features
    3. Model Training (Steps 11-13): Train baseline, tree, and XGBoost models
    4. Evaluation (Steps 14-20): Compare models and generate visualizations
    5. Advanced Analysis (Steps 21-22): Cross-validation and data quality checks

Expected Runtime: ~60 minutes
Expected Output: Model comparison results and figures in results/

Usage:
    python main.py

Requirements:
    - Python >= 3.10
    - Dependencies installed via: conda env create -f environment.yml
    - 8 GB RAM, 5 GB disk space

Author: Ricardo Guerreiro
Course: Advanced Programming 2025 - HEC Lausanne
"""

import sys
import logging
from pathlib import Path
from datetime import datetime


def main() -> None:
    """
    Execute the complete 22-step ML pipeline.
    
    This function runs all pipeline steps sequentially, from data loading
    through advanced analysis. Progress is printed to console and logged.
    
    Returns
    -------
    None
    
    Raises
    ------
    Exception
        If any critical pipeline step fails.
    """
    print("=" * 70)
    print("EARNINGS POST-ANNOUNCEMENT EXCESS RETURN PREDICTION")
    print("Advanced Programming 2025 - HEC Lausanne")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResearch Question: Can we predict 30-day excess returns")
    print("after earnings announcements using machine learning?")
    print("\nExpected Runtime: ~60 minutes")
    print("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # ====================================================================
        # PHASE 1: DATA LOADING (Steps 1-7)
        # ====================================================================
        print("\n" + "=" * 70)
        print("PHASE 1: DATA LOADING & SETUP")
        print("=" * 70)
        
        print("\n[Step 1/22] Project Setup...")
        from src.step_01_project_setup import run_step_01
        run_step_01()
        print("✓ Project setup complete")
        
        print("\n[Step 2/22] Environment Setup...")
        from src.step_02_environment_setup import run_step_02
        run_step_02()
        print("✓ Environment verified")
        
        print("\n[Step 3/22] Loading Capital IQ data...")
        from src.step_03_load_capital_iq import run_step_03
        run_step_03()
        print("✓ Capital IQ data loaded")
        
        print("\n[Step 3b/22] Extracting earnings dates...")
        from src.step_03b_extract_earnings_dates import run_step_03b
        run_step_03b()
        print("✓ Earnings dates extracted")
        
        print("\n[Step 3c/22] Loading comprehensive data...")
        from src.step_03c_load_comprehensive_data import run_step_03c
        run_step_03c()
        print("✓ Comprehensive data loaded")
        
        print("\n[Step 3d/22] Cleaning and consolidating data...")
        from src.step_03d_clean_and_consolidate import run_step_03d
        run_step_03d()
        print("✓ Data cleaned and consolidated")
        
        print("\n[Step 3e/22] Fixing fundamental extraction...")
        from src.step_03e_fix_fundamental_extraction import run_step_03e
        run_step_03e()
        print("✓ Fundamental extraction fixed")
        
        print("\n[Step 4/22] Loading SPY benchmark...")
        from src.step_04_load_spy_benchmark import run_step_04
        run_step_04()
        print("✓ SPY benchmark loaded")
        
        print("\n[Step 5/22] Fetching daily price data...")
        from src.step_05_fetch_daily_data import run_step_05
        run_step_05()
        print("✓ Daily price data fetched")
        
        print("\n[Step 6/22] Mapping earnings to daily data...")
        from src.step_06_map_earnings_to_daily import run_step_06
        run_step_06()
        print("✓ Earnings mapped to daily data")
        
        print("\n[Step 7/22] Computing target variables...")
        from src.step_07_compute_targets import run_step_07
        run_step_07()
        print("✓ Target variables computed")
        
        # ====================================================================
        # PHASE 2: FEATURE ENGINEERING (Steps 8-10)
        # ====================================================================
        print("\n" + "=" * 70)
        print("PHASE 2: FEATURE ENGINEERING")
        print("=" * 70)
        
        print("\n[Step 8/22] Building fundamental features...")
        from src.step_08_build_fundamental_features import run_step_08
        run_step_08()
        print("✓ Fundamental features built")
        
        print("\n[Step 9/22] Building market features...")
        from src.step_09_build_market_features import run_step_09
        run_step_09()
        print("✓ Market features built")
        
        print("\n[Step 10/22] Preparing modeling data...")
        from src.step_10_prepare_modelling_data import run_step_10
        run_step_10()
        print("✓ Modeling data prepared")
        
        # ====================================================================
        # PHASE 3: MODEL TRAINING (Steps 11-13)
        # ====================================================================
        print("\n" + "=" * 70)
        print("PHASE 3: MODEL TRAINING")
        print("=" * 70)
        
        print("\n[Step 11/22] Training baseline and linear models...")
        from src.step_11_train_baseline_and_linear_models import run_step_11
        run_step_11()
        print("✓ Baseline and linear models trained")
        
        print("\n[Step 12/22] Training tree-based models...")
        from src.step_12_train_tree_models import run_step_12
        run_step_12()
        print("✓ Tree-based models trained")
        
        print("\n[Step 13/22] Training XGBoost models...")
        from src.step_13_train_xgboost_models import run_step_13
        run_step_13()
        print("✓ XGBoost models trained")
        
        # ====================================================================
        # PHASE 4: EVALUATION (Steps 14-20)
        # ====================================================================
        print("\n" + "=" * 70)
        print("PHASE 4: MODEL EVALUATION")
        print("=" * 70)
        
        print("\n[Step 14/22] Comparing models...")
        from src.step_14_compare_models import run_step_14
        run_step_14()
        print("✓ Model comparison complete")
        
        print("\n[Step 15/22] Evaluating on test set...")
        from src.step_15_evaluate_on_test import run_step_15
        run_step_15()
        print("✓ Test set evaluation complete")
        
        print("\n[Step 16/22] Running advanced analysis...")
        from src.step_16_advanced_analysis import run_step_16
        run_step_16()
        print("✓ Advanced analysis complete")
        
        print("\n[Step 17/22] Generating regression figures...")
        from src.step_17_generate_regression_figures import run_step_17
        run_step_17()
        print("✓ Regression figures generated")
        
        print("\n[Step 17b/22] Generating comprehensive figures...")
        from src.step_17b_generate_comprehensive_figures import run_step_17b
        run_step_17b()
        print("✓ Comprehensive figures generated")
        
        print("\n[Step 17c/22] Generating scholarly figures...")
        from src.step_17c_generate_scholarly_figures import run_step_17c
        run_step_17c()
        print("✓ Scholarly figures generated")
        
        print("\n[Step 18/22] Building classification labels...")
        from src.step_18_build_classification_labels import run_step_18
        run_step_18()
        print("✓ Classification labels built")
        
        print("\n[Step 19/22] Training classification models...")
        from src.step_19_train_classification_models import run_step_19
        run_step_19()
        print("✓ Classification models trained")
        
        print("\n[Step 20/22] Generating classification figures...")
        from src.step_20_generate_classification_figures import run_step_20
        run_step_20()
        print("✓ Classification figures generated")
        
        # ====================================================================
        # PHASE 5: ADVANCED ANALYSIS (Steps 21-22)
        # ====================================================================
        print("\n" + "=" * 70)
        print("PHASE 5: ADVANCED ANALYSIS")
        print("=" * 70)
        
        print("\n[Step 21/22] Running cross-validation analysis...")
        from src.step_21_cross_validation_analysis import run_step_21
        run_step_21()
        print("✓ Cross-validation analysis complete")
        
        print("\n[Step 22/22] Running data quality analysis...")
        from src.step_22_data_quality_analysis import run_step_22
        run_step_22()
        print("✓ Data quality analysis complete")
        
        # ====================================================================
        # PIPELINE COMPLETE
        # ====================================================================
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {duration}")
        print("\nKey Results:")
        print("  - Regression: R² ≈ 0.0036 (no predictive power)")
        print("  - Classification: AUC ≈ 0.514 (barely above random)")
        print("  - Finding: No detectable PEAD signal in S&P 500 (2015-2024)")
        print("\nOutputs saved to:")
        print("  - results/          (figures and metrics)")
        print("  - processing/       (intermediate data)")
        print("\nNext steps:")
        print("  1. Review results in results/ directory")
        print("  2. Check cross-validation report: results/step_21/CV_ANALYSIS_REPORT.md")
        print("  3. Check data quality report: results/step_22/DATA_QUALITY_REPORT.md")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("Partial results may be available in results/ directory")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print("\nPlease check:")
        print("  1. All dependencies are installed: conda env create -f environment.yml")
        print("  2. Required data files are present in data/ directory")
        print("  3. Sufficient disk space (5 GB required)")
        print("  4. Log files in results/ for detailed error messages")
        sys.exit(1)


if __name__ == "__main__":
    main()
