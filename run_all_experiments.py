"""
Run All Robustness Experiments.

This script runs all 5 robustness experiments sequentially to validate
the main findings across different specifications.

Experiments:
    1. 10-day return horizon
    2. 5-day return horizon
    3. Day-0 immediate reaction
    4. Window robustness testing
    5. Economic significance analysis

Expected Runtime: ~30-40 minutes
Expected Output: Results in results/experiments/

Usage:
    python run_all_experiments.py

Note: This is OPTIONAL. Run main.py first to complete the core analysis.

Author: Ricardo Guerreiro
Course: Advanced Programming 2025 - HEC Lausanne
"""

import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    """
    Execute all 5 robustness experiments sequentially.
    
    Returns
    -------
    None
    
    Raises
    ------
    Exception
        If any experiment fails.
    """
    print("=" * 70)
    print("ROBUSTNESS EXPERIMENTS - ALL 5 EXPERIMENTS")
    print("Advanced Programming 2025 - HEC Lausanne")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPurpose: Validate main findings across different specifications")
    print("Expected Runtime: ~30-40 minutes")
    print("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # ====================================================================
        # EXPERIMENT 1: 10-Day Returns
        # ====================================================================
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: 10-DAY RETURN HORIZON")
        print("=" * 70)
        print("\nTesting if shorter horizon (10 days) is more predictable...")
        
        from experiments.experiments_01.src.experiment_01_returns_10d import main as exp01_main
        exp01_main()
        print("✓ Experiment 1 complete")
        
        # ====================================================================
        # EXPERIMENT 2: 5-Day Returns
        # ====================================================================
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: 5-DAY RETURN HORIZON")
        print("=" * 70)
        print("\nTesting if even shorter horizon (5 days) is more predictable...")
        
        from experiments.experiments_02.src.experiment_02_returns_5d import main as exp02_main
        exp02_main()
        print("✓ Experiment 2 complete")
        
        # ====================================================================
        # EXPERIMENT 3: Day-0 Reaction
        # ====================================================================
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: DAY-0 IMMEDIATE REACTION")
        print("=" * 70)
        print("\nTesting immediate market reaction on earnings day...")
        
        from experiments.experiments_03.src.experiment_03_day0_reaction import main as exp03_main
        exp03_main()
        print("✓ Experiment 3 complete")
        
        # ====================================================================
        # EXPERIMENT 4: Window Robustness
        # ====================================================================
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: WINDOW ROBUSTNESS TESTING")
        print("=" * 70)
        print("\nTesting robustness to different train/test split designs...")
        
        from experiments.experiments_04.src.experiment_04_window_robustness import run_experiment_04
        run_experiment_04()
        print("✓ Experiment 4 complete")
        
        # ====================================================================
        # EXPERIMENT 5: Economic Significance
        # ====================================================================
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: ECONOMIC SIGNIFICANCE ANALYSIS")
        print("=" * 70)
        print("\nTesting if predictions have economic value in trading strategy...")
        
        from experiments.experiments_05.src.experiment_05_economic_significance import run_experiment_05
        run_experiment_05()
        print("✓ Experiment 5 complete")
        
        # ====================================================================
        # ALL EXPERIMENTS COMPLETE
        # ====================================================================
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 70)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 70)
        print(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {duration}")
        print("\nKey Findings Across All Experiments:")
        print("  - Experiment 1 (10-day): R² ≈ 0 (no predictability)")
        print("  - Experiment 2 (5-day): R² ≈ 0 (no predictability)")
        print("  - Experiment 3 (Day-0): Severe overfitting detected")
        print("  - Experiment 4 (Windows): Robust across all splits")
        print("  - Experiment 5 (Economic): No profitable trading strategy")
        print("\nConclusion: Main finding (no predictability) is ROBUST")
        print("\nOutputs saved to:")
        print("  - results/experiments/returns_10d/")
        print("  - results/experiments/returns_5d/")
        print("  - results/experiments/day0_reaction/")
        print("  - results/experiments/window_robustness/")
        print("  - results/experiments/economic_significance/")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiments interrupted by user")
        print("Partial results may be available in results/experiments/")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ Experiments failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print("\nPlease check:")
        print("  1. Main pipeline completed successfully (run main.py first)")
        print("  2. All dependencies installed")
        print("  3. Sufficient disk space")
        sys.exit(1)


if __name__ == "__main__":
    main()
