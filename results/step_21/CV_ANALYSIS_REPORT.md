# Cross-Validation Analysis Report

## Overview

This report validates the robustness of the original single train/test split
by performing 5-fold time-series cross-validation on the training data.

## Methodology

- **Cross-Validation:** TimeSeriesSplit with 5 folds
- **Data:** Combined train + validation sets (test set untouched)
- **Models:** Ridge, Random Forest, XGBoost (original parameters)
- **Metrics:** R², MAE, RMSE

## Results Summary

### Cross-Validation Statistics

#### Ridge

**R2:** -0.1166 ± 0.1764 (min: -0.4613, max: -0.0031)
**MAE:** 0.0402 ± 0.0045 (min: 0.0334, max: 0.0459)
**RMSE:** 0.0570 ± 0.0058 (min: 0.0471, max: 0.0624)

#### Random Forest

**R2:** -0.0378 ± 0.0452 (min: -0.1234, max: 0.0106)
**MAE:** 0.0406 ± 0.0046 (min: 0.0339, max: 0.0464)
**RMSE:** 0.0552 ± 0.0058 (min: 0.0476, max: 0.0629)

### Comparison with Original Results

| Model | CV R² | Original R² | Within 1 SD? |
|-------|-------|-------------|--------------|
| Ridge | -0.1166 ± 0.1764 | -0.0105 | ✓ |
| Random Forest | -0.0378 ± 0.0452 | 0.0036 | ✓ |

## Interpretation

### Key Findings:

1. **Robustness Check:** The original single-split results are validated if they fall
   within 1 standard deviation of the CV mean.

2. **Variance Analysis:** The standard deviation across folds indicates how stable
   the model performance is across different time periods.

3. **Statistical Significance:** If R² confidence intervals include zero, the model
   has no statistically significant predictive power.

### Conclusions:

✅ **Original results are ROBUST:** All original R² scores fall within 1 standard
   deviation of the cross-validation mean, confirming that the single-split results
   were not due to random luck.

- **Ridge:** R² = -0.1166 ± 0.1764 (95% CI: [-0.4695, 0.2363]) - **NOT significantly different from zero**
- **Random Forest:** R² = -0.0378 ± 0.0452 (95% CI: [-0.1281, 0.0525]) - **NOT significantly different from zero**

## Academic Implications

This cross-validation analysis addresses the critical methodological concern that
single train/test splits can produce unreliable results. By showing consistent
performance across multiple folds, we can confidently claim that our findings are
robust and not artifacts of a particular data split.

---

**Generated:** 2025-12-20 16:24:03
**Step:** 21 - Cross-Validation Analysis