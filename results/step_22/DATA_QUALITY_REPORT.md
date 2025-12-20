# Data Quality Analysis Report

## Executive Summary

**Analysis Date:** 2025-12-20 16:28:04
**Total Samples:** 19,300
**Total Features:** 21

---

## 1. Missing Data Analysis

### Overall Statistics

- **Total cells:** 405,300
- **Missing cells:** 94,008 (23.19%)
- **Complete cases:** 9,586 (49.67%)
- **Incomplete cases:** 9,714 (50.33%)

### Key Findings

⚠️ **WARNING:** 23.2% of data is missing

### Top 10 Features with Most Missing Data

| Feature | Missing Count | Missing % |
|---------|--------------|-----------|
| f_eps_surprise_pct | 8,287 | 42.94% |
| f_eps_surprise_value | 8,258 | 42.79% |
| f_revenue_growth_yoy | 7,961 | 41.25% |
| f_revenue_growth_qoq | 7,357 | 38.12% |
| f_asset_turnover | 7,160 | 37.10% |
| f_net_margin | 7,160 | 37.10% |
| f_operating_margin | 7,160 | 37.10% |
| f_net_income_change_yoy | 6,473 | 33.54% |
| f_net_income_change_qoq | 5,799 | 30.05% |
| f_cash_to_assets_ratio | 5,783 | 29.96% |

---

## 2. Outlier Analysis

### Z-Score Method (|z| > 3)

**Total outliers detected:** 4,784

**Top 5 features:**

- f_cash_to_assets_ratio: 467 (2.42%)
- f_asset_turnover: 450 (2.33%)
- f_net_income_change_yoy: 389 (2.02%)
- f_roa: 355 (1.84%)
- pre_volatility_30d: 328 (1.70%)

### IQR Method (1.5 * IQR)

**Total outliers detected:** 26,620

**Top 5 features:**

- f_net_income_change_qoq: 3,329 (17.25%)
- f_revenue_growth_qoq: 2,951 (15.29%)
- f_net_income_change_yoy: 2,723 (14.11%)
- f_cashflow_proxy_ffo_capex: 2,254 (11.68%)
- pre_avg_volume_30d: 2,060 (10.67%)

### ⚠️ Extreme Values Detected

The following features contain extreme values (>1e6 or <-1e6):

- **f_cashflow_proxy_ffo_capex**: min=-1.68e+07, max=1.24e+08
- **f_eps_surprise_pct**: min=-8.86e+03, max=3.30e+06
- **f_net_income_change_qoq**: min=-3.61e+07, max=3.73e+07
- **f_net_income_change_yoy**: min=-3.61e+07, max=3.73e+07
- **pre_avg_volume_30d**: min=0.00e+00, max=1.12e+09

---

## 3. Data Leakage Verification

✅ **No critical data leakage issues detected**

**Summary:**
- Critical issues: 0
- Warning issues: 2

### Issues Detected:

⚠️ **future_information** (WARNING)

⚠️ **quarter_alignment** (WARNING)

---

## 4. Feature Distribution Analysis

⚠️ **Found 13 highly skewed features (|skew| > 3)**

- f_eps_surprise_pct: skewness = 104.93
- f_roe: skewness = 45.41
- f_revenue_growth_yoy: skewness = 19.41
- f_revenue_growth_qoq: skewness = 17.17
- pre_avg_volume_30d: skewness = 14.78
- f_eps_surprise_value: skewness = 11.39
- f_net_margin: skewness = -10.50
- f_cashflow_proxy_ffo_capex: skewness = 7.12
- f_operating_margin: skewness = -6.05
- f_net_income_change_qoq: skewness = 4.59

---

## 5. Recommendations

### High Priority

### Medium Priority

4. **Investigate extreme values:** Verify these are not data errors

5. **Consider transforming skewed features:** Log or Box-Cox transformations

### Low Priority

---

## 6. Conclusion

✅ **No critical data quality issues detected.**

While there are some warnings (missing data, outliers), the data quality
is acceptable for modeling. The main concerns are:

- 23.2% missing data (being handled by imputation)
- 4,784 outliers detected (may be legitimate extreme values)

**Your current preprocessing pipeline appears adequate.**

---

**Generated:** 2025-12-20 16:28:04
**Step:** 22 - Data Quality Analysis
**Status:** Analysis complete (no data modified)