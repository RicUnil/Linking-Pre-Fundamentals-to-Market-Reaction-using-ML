# Earnings Post-Announcement Excess Return Prediction

**Advanced Programming 2025 — HEC Lausanne**  
**Ricardo Guerreiro | MSc Finance**

---

## Overview

**Research Question:** Can we predict 30-day excess returns after earnings announcements using machine learning?

**Main Finding:** No detectable predictive power (R² ≈ 0, AUC ≈ 0.50)

**Data:** 19,300 earnings events | S&P 500 | 2015–2024 | 21 features

---

## Project Structure

```
windsurf-project/
├── main.py                    # Single entry point - runs entire pipeline
├── run_all_experiments.py     # Optional: runs all 5 experiments
├── src/                       # 22-step modular pipeline
│   ├── step_01_project_setup.py
│   ├── step_02_environment_setup.py
│   ├── ...
│   ├── step_22_data_quality_analysis.py
│   ├── config.py              # Configuration settings
│   ├── models/                # Model implementations
│   ├── preprocessing/         # Feature engineering
│   ├── evaluation/            # Metrics and validation
│   ├── analysis/              # Statistical tests
│   └── visualization/         # Plotting functions
├── experiments/               # 5 robustness experiments (optional)
│   ├── experiments_01/        # 10-day returns
│   ├── experiments_02/        # 5-day returns
│   ├── experiments_03/        # Day-0 reaction
│   ├── experiments_04/        # Window robustness
│   └── experiments_05/        # Economic significance
├── data/                      # Raw data files
│   ├── RAW_DATA.csv
│   ├── BENCHMARK.csv
│   └── Quarter_*.csv
├── results/                   # Generated outputs (figures, metrics)
├── tests/                     # Unit tests
├── notebooks/                 # Analysis notebooks
├── environment.yml            # Conda dependencies
├── requirements.txt           # pip dependencies
├── PROPOSAL.md                # Project proposal
├── AI_USAGE.md                # AI tool usage documentation
└── README.md                  # This file
```

**Pipeline Overview (22 Steps):**
1. **Steps 1-7:** Data loading and validation
2. **Steps 8-10:** Feature engineering and dataset creation
3. **Steps 11-13:** Model training (Ridge, Random Forest, XGBoost)
4. **Steps 14-20:** Evaluation and visualization
5. **Steps 21-22:** Cross-validation and data quality checks

---

## Setup Instructions

### Requirements
- Python ≥ 3.10
- 8 GB RAM
- 5 GB disk space

### Option 1: Conda (Recommended for Nuvolos)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate earnings-env
```

### Option 2: pip + venv (Local development)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Main Pipeline (Required)

```bash
# Run complete 22-step pipeline
python main.py
```

**What it does:**
- Loads and validates data (Steps 1-7)
- Engineers 21 features (Steps 8-10)
- Trains 4 models: Ridge, Random Forest, XGBoost, Gradient Boosting (Steps 11-13)
- Evaluates models and generates figures (Steps 14-20)
- Performs cross-validation and data quality checks (Steps 21-22)

**Output:**
- Results printed to console
- Figures saved to `results/` directory
- Metrics saved as JSON/CSV files


### Experiments (Optional)

```bash
# Run all 5 experiments at once
python run_all_experiments.py
```

**Or run individually:**
```bash
python3 -m experiments.experiments_01.src.experiment_01_returns_10d
python3 -m experiments.experiments_02.src.experiment_02_returns_5d
python3 -m experiments.experiments_03.src.experiment_03_day0_reaction
python3 -m experiments.experiments_04.src.experiment_04_window_robustness
python3 -m experiments.experiments_05.src.experiment_05_economic_significance
```


### Run Individual Steps (Advanced)

```bash
# Run specific steps manually
python3 -m src.step_01_project_setup
python3 -m src.step_10_create_final_dataset
python3 -m src.step_16_advanced_analysis
# etc.
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test R²** | 0.0036 | <1% variance explained |
| **Test AUC** | 0.514 | Barely above random (0.50) |
| **CV R²** | -0.038 ± 0.045 | 95% CI includes zero |

**Statistical Validation:**
- Bootstrap confidence intervals: All include zero
- Permutation tests: p > 0.05 (not significant)
- Robust across sectors and market regimes

**Conclusion:** No evidence of predictability in our setting

---

## Methodology

**Features (21 total):**
- Fundamental: Earnings surprise, revenue growth, ROE, P/E ratio, leverage
- Market: Pre-announcement returns (1d, 5d, 30d), volatility, volume
- Momentum: Stock returns (1m, 3m, 6m)

**Models:**
- Baseline: Mean predictor, CAPM
- Linear: Ridge regression
- Tree-based: Random Forest, Gradient Boosting
- Boosting: XGBoost

**Evaluation:**
- Temporal train/val/test split (no look-ahead bias)
- 5-fold time-series cross-validation
- Bootstrap confidence intervals
- Permutation tests for significance

---

## Documentation

- **[PROPOSAL.md](PROPOSAL.md)** - Project proposal (300-500 words)
- **[AI_USAGE.md](AI_USAGE.md)** - AI tool usage transparency
- **[notebooks/](notebooks/)** - Jupyter analysis notebooks

---

## Author

**Ricardo Contente Guerreiro**  
MSc Finance | HEC Lausanne  
Advanced Programming 2025

---

## License

Academic project for educational purposes.
