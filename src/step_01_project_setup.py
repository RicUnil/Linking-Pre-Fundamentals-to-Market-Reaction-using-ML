"""
Step 01 — Project Setup.

This step initializes logging, verifies folder structure, and prepares the project
environment for the remaining steps of the ML pipeline.

The script performs the following tasks:
1. Initialize logging system
2. Verify and create required folder structure
3. Create step-specific results directory
4. Save completion marker file
5. Validate configuration settings

Usage
-----
    python -m src.step_01_project_setup
"""

import logging
from pathlib import Path
from typing import NoReturn
from datetime import datetime

from src.config import settings


def run_step_01() -> None:
    """
    Execute Step 01: Project Setup.
    
    This function initializes the project environment by:
    - Setting up logging
    - Creating required directories
    - Validating configuration
    - Saving a completion marker
    
    Returns
    -------
    None
    
    Raises
    ------
    Exception
        If any critical setup step fails.
    """
    # Initialize logging
    logger = settings.setup_logging("step_01_project_setup")
    logger.info("=" * 70)
    logger.info("STEP 01: PROJECT SETUP")
    logger.info("=" * 70)
    
    try:
        # Step 1: Ensure all required directories exist
        logger.info("Verifying project folder structure...")
        settings.ensure_directories()
        logger.info(f"✓ Project root: {settings.PROJECT_ROOT}")
        logger.info(f"✓ Data directory: {settings.DATA_DIR}")
        logger.info(f"✓ Results directory: {settings.RESULTS_DIR}")
        logger.info(f"✓ Notebooks directory: {settings.NOTEBOOKS_DIR}")
        logger.info(f"✓ Tests directory: {settings.TESTS_DIR}")
        logger.info(f"✓ Report directory: {settings.REPORT_DIR}")
        logger.info(f"✓ Experiments directory: {settings.EXPERIMENTS_DIR}")
        
        # Step 2: Create step-specific results directory
        logger.info("\nCreating Step 01 results directory...")
        step_results_dir = settings.get_step_results_dir(1)
        logger.info(f"✓ Step 01 results directory: {step_results_dir}")
        
        # Step 3: Validate configuration settings
        logger.info("\nValidating configuration settings...")
        logger.info(f"✓ Random seed: {settings.RANDOM_SEED}")
        logger.info(f"✓ Train/test split date: {settings.TRAIN_TEST_SPLIT_DATE}")
        logger.info(f"✓ Benchmark ticker: {settings.SPY_TICKER}")
        
        # Step 4: Verify data files exist
        logger.info("\nVerifying data files...")
        expected_files = [
            "RAW_DATA.csv",
            "BENCHMARK.csv",
            "Quarter_1.csv",
            "Quarter_2.csv",
            "Quarter_3.csv",
            "Quarter_4.csv"
        ]
        
        for filename in expected_files:
            file_path = settings.DATA_DIR / filename
            if file_path.exists():
                logger.info(f"✓ Found: {filename}")
            else:
                logger.warning(f"⚠ Missing: {filename} (will be used in later steps)")
        
        # Step 5: Save completion marker
        logger.info("\nSaving completion marker...")
        completion_file = step_results_dir / "step_01_completed.txt"
        
        completion_message = f"""Step 01 - Project Setup
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Project Configuration:
- Project Root: {settings.PROJECT_ROOT}
- Data Directory: {settings.DATA_DIR}
- Results Directory: {settings.RESULTS_DIR}
- Random Seed: {settings.RANDOM_SEED}
- Train/Test Split Date: {settings.TRAIN_TEST_SPLIT_DATE}
- Benchmark Ticker: {settings.SPY_TICKER}

Status: SUCCESS
All required directories have been created and verified.
The project is ready for subsequent pipeline steps.
"""
        
        with open(completion_file, 'w') as f:
            f.write(completion_message)
        
        logger.info(f"✓ Completion marker saved: {completion_file}")
        
        # Final success message
        logger.info("\n" + "=" * 70)
        logger.info("Step 01 completed successfully")
        logger.info("=" * 70)
        logger.info("\nThe project environment is now ready.")
        logger.info("You may proceed to Step 02 when ready.")
        
    except Exception as e:
        logger.error(f"\n✗ Step 01 failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    run_step_01()
