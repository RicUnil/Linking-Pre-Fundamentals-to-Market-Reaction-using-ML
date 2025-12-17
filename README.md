# Earnings Post-Announcement Excess Return Prediction

**Advanced Programming 2025 — HEC Lausanne**  
**Ricardo Guerreiro | MSc Finance**

---

## 🎯 Overview

Testing whether post-earnings excess returns are predictable using machine learning.

**Research Question:** Can we predict 30-day excess returns after earnings announcements?

**Main Finding:** **No detectable predictive power in our setting** (R² ≈ 0, AUC ≈ 0.50)
- Results consistent across models, time horizons, and validation methods
- Findings consistent with rapid price incorporation for S&P 500 large-caps (2015–2024)
- Results conditional on our feature set, universe, and evaluation design
- See [Limitations](#️-limitations) for scope and interpretation

---

## 📊 Key Results

| Analysis | Best Model | Test Metric | Result |
|----------|------------|-------------|--------|
| **Regression** | Random Forest | R² = 0.0036 | <1% variance explained |
| **Classification** | Gradient Boosting | AUC = 0.514 | Barely above random |
| **Cross-Validation** | Random Forest | R² = -0.038 ± 0.045 | 95% CI includes zero |
| **Multi-Horizon** | All models | R² ≈ 0 | No PEAD detected |

**Statistical Tests:**
- Bootstrap CIs: All include zero
- Permutation tests: p > 0.05 (not significant)
- Robustness: Holds across sectors and market regimes

**Data Quality:**
- 19,300 earnings events (S&P 500, 2015–2024)
- 23.2% missing data (median imputation)
- No critical data leakage detected

---

## 📊 Data

**Dataset:** 19,300 earnings announcements | S&P 500 | 2015–2024  
**Features:** 21 (fundamental + market + momentum)  
**Split:** Train 50.8% | Val 12.7% | Test 36.5% (temporal)

---

## 📂 Project Structure

```
windsurf-project/
├── src/                       # 27 pipeline steps (~21,800 lines)
│   ├── step_01-22_*.py        # Main pipeline
│   ├── analysis/              # Statistical tests
│   ├── preprocessing/         # Feature engineering
│   └── visualization/         # Plotting
├── experiments/               # Robustness experiments
│   ├── experiments_01/        # 10-day returns
│   ├── experiments_02/        # 5-day returns
│   ├── experiments_03/        # 0-day reaction
│   ├── experiments_04/        # Window robustness
│   └── experiments_05/        # Economic significance
├── tests/                     # Unit tests (pytest)
├── results/                   # Outputs (~500 MB, 30+ figures)
├── data/                      # Raw data files
└── docs/                      # Documentation
```

---

## ⚙️ Setup

**Requirements:** Python ≥ 3.10 | 8 GB RAM | 5 GB disk

```bash
# Install dependencies
pip install -r requirements.txt

# Key packages: pandas, scikit-learn, xgboost, shap, matplotlib
```

---

## 🚀 Usage

### Run Complete Pipeline (~60 min)

```bash
# Run steps 1-22 sequentially
python3 -m src.step_01_project_setup
python3 -m src.step_02_environment_setup
# ... (see full list in original README or run_all.sh)
python3 -m src.step_22_data_quality_analysis
```

**Pipeline Phases:**
1. Data Loading (Steps 1-7) → 15-30 min
2. Feature Engineering (Steps 8-10) → 5-10 min  
3. Model Training (Steps 11-13) → 10-20 min
4. Evaluation (Steps 14-20) → 15-25 min
5. Advanced Analysis (Steps 16, 21-22) → 15-30 min

### Run Experiments

```bash
# Multi-horizon experiments (5-10 min each)
python3 -m experiments.experiments_01.src.experiment_01_returns_10d
python3 -m experiments.experiments_02.src.experiment_02_returns_5d
python3 -m experiments.experiments_03.src.experiment_03_day0_reaction
python3 -m experiments.experiments_04.src.experiment_04_window_robustness
python3 -m experiments.experiments_05.src.experiment_05_economic_significance
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 🔬 Methodology

**Pipeline:**
1. Data collection (earnings, prices, fundamentals)
2. Feature engineering (21 features: fundamental + market + momentum)
3. Target: 30-day excess return vs SPY
4. Temporal train/val/test split (no look-ahead)
5. Model training (Ridge, Random Forest, XGBoost, Gradient Boosting)
6. Evaluation (R², AUC, bootstrap CIs, permutation tests)
7. Advanced analysis (SHAP, sector/regime robustness)

**Robustness:**
- 5-fold time-series cross-validation
- Multi-horizon experiments (0, 5, 10, 30 days)
- Window robustness testing (quarterly vs yearly)
- Economic significance assessment
- Data quality verification

---

## ⚠️ Limitations & Scope

### What Our Evidence Supports ✅
- **No detectable predictive power** within our specific experimental setup
- Null result holds across multiple models, validation methods, and robustness checks
- Findings are **consistent with** (but do not prove) rapid price incorporation for large-cap equities

### What Our Evidence Does NOT Support ❌
- Universal claims about market efficiency or impossibility of prediction
- Conclusions beyond our feature set (21 variables), universe (S&P 500), or horizon (30 days)
- Economic profitability assessment (no transaction costs included)

### Methodological Scope

Our findings are conditional on deliberate design choices:

1. **Feature Set:** 21 fundamental/market variables (excludes textual data, analyst forecasts, options signals)
2. **Universe:** S&P 500 large-caps only (survivorship bias, limited to liquid stocks)
3. **Time Period:** 2015–2024 (specific market regime, may not generalize)
4. **Prediction Horizon:** 30-day returns (different horizons may show different patterns)
5. **Temporal Structure:** Fixed train/val/test splits with robustness checks (Experiment 4)

**Key Insight:** Our null result is informative—it demonstrates that even sophisticated ML models fail to extract signal from post-earnings fundamental data in our setting. This is a meaningful empirical finding about the limits of predictability, not a methodological failure.

**Precise Interpretation:** *"We find no evidence of predictability of 30-day post-earnings excess returns within our feature set, universe, horizon, and evaluation design."*

📄 **See [`INTERPRETATION_AND_LIMITATIONS.md`](INTERPRETATION_AND_LIMITATIONS.md) for comprehensive academic discussion.**

---

## 📚 Documentation

- **[INTERPRETATION_AND_LIMITATIONS.md](INTERPRETATION_AND_LIMITATIONS.md)** - Academic interpretation and scope discussion
- **[LIMITATIONS.md](LIMITATIONS.md)** - Comprehensive limitations and threats to validity
- **[AI_USAGE.md](AI_USAGE.md)** - AI tool usage transparency
- **[experiments/EXPERIMENTS_COMPARISON.md](experiments/EXPERIMENTS_COMPARISON.md)** - Multi-horizon results
- **[notebooks/](notebooks/)** - Jupyter analysis notebooks
- **[results/step_21/CV_ANALYSIS_REPORT.md](results/step_21/CV_ANALYSIS_REPORT.md)** - Cross-validation analysis
- **[results/step_22/DATA_QUALITY_REPORT.md](results/step_22/DATA_QUALITY_REPORT.md)** - Data quality verification

---

## 🎯 Highlights

**Technical:**
- 21,800 lines of Python | 27 modular steps | 5 experiments
- Advanced analysis: Bootstrap CIs, permutation tests, SHAP
- 30+ publication-quality figures

**Academic Rigor:**
- Hypothesis testing with proper statistical methods
- 5-fold time-series cross-validation
- Data quality verification (missing data, outliers, leakage)
- Multiple testing corrections (Bonferroni, FDR)
- Honest reporting of negative results

**Key Insights:**
- No detectable predictive power in our setting (R² ≈ 0, AUC ≈ 0.50)
- Findings consistent with rapid price incorporation for S&P 500 large-caps
- Null result robust across models, horizons, and validation methods
- Results conditional on our specific feature set, universe, and evaluation design

---

## 🤖 AI Usage

- **ChatGPT 4o:** Strategic planning
- **Claude Sonnet 3.5:** Code implementation  
- **Human:** 100% research design, analysis, interpretation

See [AI_USAGE.md](AI_USAGE.md) for full transparency.

---

## 👤 Author

**Ricardo Guerreiro**  
MSc Finance  
HEC Lausanne  
Advanced Programming 2025

---

## 📜 License

Academic project for educational purposes.  
All rights reserved.
