"""
Step 02 — Environment and Initial Documentation Setup.

This step creates and verifies the project's development environment files
and initial documentation structure. It ensures that all necessary configuration
files, dependency specifications, and documentation templates are in place.

The script performs the following tasks:
1. Verify Python version compatibility
2. Create/verify requirements.txt with pinned dependencies
3. Create/verify .gitignore with appropriate patterns
4. Create/verify README.md with project structure
5. Create/verify environment.yml for Conda users
6. Create/verify report/PROPOSAL.md with proposal skeleton
7. Save completion marker

Usage
-----
    python -m src.step_02_environment_setup
"""

import logging
import sys
from pathlib import Path
from typing import NoReturn
from datetime import datetime

from src.config import settings


def run_step_02() -> None:
    """
    Execute Step 02: Environment and Documentation Setup.
    
    This function sets up the development environment by creating or verifying
    all necessary configuration files, dependency specifications, and initial
    documentation templates.
    
    Returns
    -------
    None
    
    Raises
    ------
    Exception
        If any critical setup step fails.
    """
    # Initialize logging
    logger = settings.setup_logging("step_02_environment_setup")
    logger.info("=" * 70)
    logger.info("STEP 02: ENVIRONMENT AND DOCUMENTATION SETUP")
    logger.info("=" * 70)
    
    try:
        # Ensure directories exist
        settings.ensure_directories()
        
        # Step 1: Verify Python version
        logger.info("\nVerifying Python version...")
        python_version = sys.version_info
        logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
            logger.warning("⚠ Python version is below 3.10. Consider upgrading for best compatibility.")
        else:
            logger.info("✓ Python version is compatible (>= 3.10)")
        
        # Step 2: Create/verify requirements.txt
        logger.info("\nCreating/verifying requirements.txt...")
        requirements_file = settings.PROJECT_ROOT / "requirements.txt"
        
        if not requirements_file.exists():
            requirements_content = """# Requirements for the Advanced Programming 2025 earnings prediction project.
# The student may adjust versions if necessary.

pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
yfinance==0.2.33
matplotlib==3.8.2
seaborn==0.13.0
pyyaml==6.0.1
pytest==7.4.3
joblib==1.3.2
"""
            with open(requirements_file, 'w') as f:
                f.write(requirements_content)
            logger.info(f"✓ Created requirements.txt")
        else:
            logger.info(f"✓ requirements.txt already exists")
        
        # Step 3: Create/verify .gitignore
        logger.info("\nCreating/verifying .gitignore...")
        gitignore_file = settings.PROJECT_ROOT / ".gitignore"
        
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
.venv/
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
results/
experiments/
*.log

# Environment variables
.env
.env.local

# Distribution / packaging
build/
dist/
*.egg-info/
"""
        
        if not gitignore_file.exists():
            with open(gitignore_file, 'w') as f:
                f.write(gitignore_content)
            logger.info(f"✓ Created .gitignore")
        else:
            # Append missing patterns if needed
            existing_content = gitignore_file.read_text()
            if "__pycache__" not in existing_content:
                with open(gitignore_file, 'a') as f:
                    f.write("\n" + gitignore_content)
                logger.info(f"✓ Updated .gitignore with missing patterns")
            else:
                logger.info(f"✓ .gitignore already exists")
        
        # Step 4: Create/verify README.md
        logger.info("\nCreating/verifying README.md...")
        readme_file = settings.PROJECT_ROOT / "README.md"
        
        readme_content = """# Earnings Post-Announcement Excess Return Prediction

**Advanced Programming 2025 — HEC Lausanne**

## Project Overview

This project aims to predict the 30-day post-earnings excess return of stocks relative to the SPY benchmark (S&P 500 ETF). The analysis includes both regression (continuous excess return) and binary classification (outperform vs. underperform) approaches.

### Research Question

Can we predict post-earnings announcement excess returns using fundamental and market data?

**Null Hypothesis (H₀):** Post-earnings excess returns are unpredictable.

## Data Description

The project uses the following datasets:

- `data/RAW_DATA.csv` — Main earnings dataset (Capital IQ style)
- `data/BENCHMARK.csv` — SPY benchmark data
- `data/Quarter_1.csv` — Q1 quarterly information
- `data/Quarter_2.csv` — Q2 quarterly information
- `data/Quarter_3.csv` — Q3 quarterly information
- `data/Quarter_4.csv` — Q4 quarterly information

## Project Structure

```
windsurf-project/
├── data/                   # Raw data files
├── src/                    # Source code (20-step pipeline)
│   ├── step_01_*.py
│   ├── step_02_*.py
│   └── ...
├── tests/                  # Unit tests
├── results/                # Step outputs and artifacts
├── experiments/            # Experimental runs
├── notebooks/              # Jupyter notebooks for analysis
├── report/                 # Project reports and documentation
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment specification
└── README.md              # This file
```

## Environment Setup

### Option 1: Using pip (recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\\Scripts\\activate    # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate earnings-env
```

## Running the Pipeline

The project is organized as a 20-step modular pipeline. Each step is executed independently:

```bash
# Step 01: Project setup
python -m src.step_01_project_setup

# Step 02: Environment setup
python -m src.step_02_environment_setup

# Future steps will be added incrementally
# python -m src.step_03_*
# ...
```

## Methodology

1. **Data Loading and Validation** — Load and validate all datasets
2. **Feature Engineering** — Create fundamental and market-based features
3. **Target Construction** — Calculate 30-day excess returns vs. SPY
4. **Train/Test Split** — Temporal split (cutoff: 2020-01-01)
5. **Baseline Models** — Historical mean, CAPM
6. **ML Models** — Regression and classification approaches
7. **Evaluation** — Rolling out-of-sample validation
8. **Hypothesis Testing** — Statistical tests for H₀

## Requirements

- Python >= 3.10
- See `requirements.txt` for package dependencies

## Project Guidelines

- **Code Style:** PEP8 compliant
- **Documentation:** NumPy-style docstrings
- **Testing:** Unit tests for critical functions
- **Reproducibility:** Fixed random seed (42)
- **Modularity:** Clean separation of concerns

## Authors

HEC Lausanne — Advanced Programming 2025

## License

Academic project for educational purposes.
"""
        
        if not readme_file.exists():
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            logger.info(f"✓ Created README.md")
        else:
            # Check if README has environment setup section
            existing_content = readme_file.read_text()
            if "Environment Setup" not in existing_content:
                # Append environment setup section
                with open(readme_file, 'a') as f:
                    f.write("\n\n## Environment Setup\n\n")
                    f.write("```bash\n")
                    f.write("python -m venv .venv\n")
                    f.write("source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows\n")
                    f.write("pip install -r requirements.txt\n")
                    f.write("```\n")
                logger.info(f"✓ Updated README.md with environment setup")
            else:
                logger.info(f"✓ README.md already exists")
        
        # Step 5: Create/verify environment.yml
        logger.info("\nCreating/verifying environment.yml...")
        env_file = settings.PROJECT_ROOT / "environment.yml"
        
        env_content = """name: earnings-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - pytest
  - pip
  - pip:
      - yfinance
      - joblib
      - pyyaml
"""
        
        if not env_file.exists():
            with open(env_file, 'w') as f:
                f.write(env_content)
            logger.info(f"✓ Created environment.yml")
        else:
            logger.info(f"✓ environment.yml already exists")
        
        # Step 6: Create/verify report/PROPOSAL.md
        logger.info("\nCreating/verifying report/PROPOSAL.md...")
        report_dir = settings.REPORT_DIR
        report_dir.mkdir(parents=True, exist_ok=True)
        
        proposal_file = report_dir / "PROPOSAL.md"
        
        proposal_content = """# Project Proposal — Advanced Programming 2025

## 1. Project Title

**Predicting Post-Earnings Announcement Excess Returns: A Machine Learning Approach**

## 2. Motivation and Context

Earnings announcements are among the most significant events in financial markets, often triggering substantial price movements. Understanding whether these movements are predictable can provide insights into market efficiency and inform trading strategies.

This project investigates whether machine learning models can predict the 30-day excess return of stocks following earnings announcements, relative to the SPY (S&P 500 ETF) benchmark.

## 3. Research Question and Null Hypothesis

**Research Question:**  
Can we predict post-earnings announcement excess returns using fundamental financial data and market indicators?

**Null Hypothesis (H₀):**  
Post-earnings excess returns (relative to SPY) are unpredictable and follow a random distribution.

**Alternative Hypothesis (H₁):**  
Post-earnings excess returns exhibit predictable patterns that can be captured by machine learning models.

## 4. Data Description

### Primary Datasets

- **RAW_DATA.csv** — Main earnings dataset containing:
  - Company identifiers
  - Earnings announcement dates
  - Financial fundamentals (Capital IQ style)
  - Historical price data

- **BENCHMARK.csv** — SPY benchmark data:
  - Quarterly SPY returns
  - Market performance metrics

- **Quarter_1.csv – Quarter_4.csv** — Quarterly supplementary data:
  - Additional fundamental metrics
  - Sector classifications
  - Market conditions

### Target Variable

- **30-day excess return** = (Stock return over 30 days post-earnings) - (SPY return over same period)
- **Binary label** = 1 if excess return > 0 (outperform), 0 otherwise (underperform)

## 5. Methodology Overview

### 5.1 Data Pipeline

1. **Data Loading and Validation** — Import and validate all datasets
2. **Data Cleaning** — Handle missing values, outliers, and inconsistencies
3. **Feature Engineering** — Create predictive features from fundamentals and market data
4. **Target Construction** — Calculate 30-day excess returns

### 5.2 Feature Categories

- **Fundamental Features:** EPS surprise, revenue growth, profit margins, leverage ratios
- **Market Features:** Pre-announcement volatility, trading volume, momentum indicators
- **Temporal Features:** Quarter, year, days since last earnings
- **Sector Features:** Industry classification, sector performance

### 5.3 Modeling Approach

**Baseline Models:**
- Historical mean excess return
- CAPM-based prediction
- Naive classifier (majority class)

**Machine Learning Models:**
- Linear Regression (with regularization)
- Random Forest (regression and classification)
- Gradient Boosting (XGBoost/LightGBM)
- Logistic Regression (for classification)

### 5.4 Evaluation Strategy

- **Temporal Train/Test Split:** Data before 2020-01-01 for training, after for testing
- **Rolling Window Validation:** Simulate real-world deployment
- **Regression Metrics:** RMSE, MAE, R²
- **Classification Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Statistical Tests:** t-tests, permutation tests for H₀

## 6. Expected Contributions and Limitations

### Expected Contributions

- Rigorous test of market efficiency in the post-earnings context
- Identification of most predictive features for excess returns
- Comparison of ML approaches vs. traditional financial models
- Reproducible, modular pipeline for financial ML research

### Limitations

- Limited to available fundamental data (no alternative data sources)
- Survivorship bias in historical data
- Transaction costs and market impact not modeled
- Model performance may degrade in regime changes

## 7. Timeline

| Week | Tasks |
|------|-------|
| 1-2  | Data loading, cleaning, and exploratory analysis |
| 3-4  | Feature engineering and target construction |
| 5-6  | Baseline model implementation and evaluation |
| 7-8  | ML model development and hyperparameter tuning |
| 9-10 | Out-of-sample testing and hypothesis testing |
| 11-12| Final report writing and presentation preparation |

## 8. Technical Requirements

- **Language:** Python 3.10+
- **Key Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Code Standards:** PEP8, type hints, NumPy docstrings
- **Testing:** pytest for unit tests
- **Reproducibility:** Fixed random seed (42), version control

## 9. Deliverables

1. Complete 20-step modular pipeline
2. Comprehensive test suite
3. Final report with findings and visualizations
4. Presentation slides
5. Reproducible code repository

---

**Project Start Date:** December 2025  
**Course:** Advanced Programming 2025 — HEC Lausanne
"""
        
        if not proposal_file.exists():
            with open(proposal_file, 'w') as f:
                f.write(proposal_content)
            logger.info(f"✓ Created report/PROPOSAL.md")
        else:
            logger.info(f"✓ report/PROPOSAL.md already exists")
        
        # Step 7: Create step results directory and completion marker
        logger.info("\nCreating Step 02 results directory...")
        step_results_dir = settings.get_step_results_dir(2)
        logger.info(f"✓ Step 02 results directory: {step_results_dir}")
        
        # Save completion marker
        logger.info("\nSaving completion marker...")
        completion_file = step_results_dir / "step_02_completed.txt"
        
        completion_message = f"""Step 02 - Environment and Documentation Setup
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Files Created/Verified:
- requirements.txt (Python dependencies)
- .gitignore (Version control ignore patterns)
- README.md (Project documentation)
- environment.yml (Conda environment specification)
- report/PROPOSAL.md (Project proposal)

Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}

Status: SUCCESS
All environment and documentation files are in place.
The project is ready for data loading and processing steps.
"""
        
        with open(completion_file, 'w') as f:
            f.write(completion_message)
        
        logger.info(f"✓ Completion marker saved: {completion_file}")
        
        # Final success message
        logger.info("\n" + "=" * 70)
        logger.info("Step 02 completed successfully")
        logger.info("=" * 70)
        logger.info("\nEnvironment and basic documentation are ready.")
        logger.info("You may proceed to Step 03 when ready.")
        
    except Exception as e:
        logger.error(f"\n✗ Step 02 failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    run_step_02()
