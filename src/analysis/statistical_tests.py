"""
Statistical Rigor Module

This module provides comprehensive statistical testing for model evaluation:
- Hypothesis testing (t-tests, F-tests)
- Bootstrap confidence intervals for R²
- Multiple testing corrections (Bonferroni, FDR)
- Permutation tests for feature importance

Author: Academic Project Enhancement
Date: December 2025
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils import resample


class StatisticalTester:
    """
    Comprehensive statistical testing for model evaluation.
    
    Provides rigorous hypothesis testing and confidence intervals
    to validate model performance beyond simple metrics.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize statistical tester.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def test_model_vs_baseline(
        self,
        y_true: np.ndarray,
        y_pred_model: np.ndarray,
        y_pred_baseline: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Test if model significantly outperforms baseline using paired t-test.
        
        H₀: Model and baseline have equal performance
        H₁: Model outperforms baseline
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values
        y_pred_model : np.ndarray
            Model predictions
        y_pred_baseline : np.ndarray
            Baseline predictions
        alpha : float
            Significance level (default: 0.05)
            
        Returns
        -------
        Dict containing:
            - t_statistic: t-test statistic
            - p_value: p-value (one-tailed)
            - significant: whether difference is significant
            - mean_diff: mean difference in squared errors
            - effect_size: Cohen's d effect size
        """
        # Compute squared errors
        se_model = (y_true - y_pred_model) ** 2
        se_baseline = (y_true - y_pred_baseline) ** 2
        
        # Difference (negative means model is better)
        diff = se_model - se_baseline
        
        # Paired t-test (one-tailed: model < baseline)
        t_stat, p_value_two_tailed = stats.ttest_rel(se_model, se_baseline)
        p_value = p_value_two_tailed / 2  # One-tailed
        
        # If t_stat > 0, model is worse, so p_value = 1 - p_value
        if t_stat > 0:
            p_value = 1 - p_value
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'mean_diff': float(mean_diff),
            'effect_size': float(cohens_d),
            'interpretation': self._interpret_t_test(p_value, mean_diff, alpha)
        }
    
    def _interpret_t_test(self, p_value: float, mean_diff: float, alpha: float) -> str:
        """Interpret t-test results."""
        if p_value < alpha:
            if mean_diff < 0:
                return f"Model significantly outperforms baseline (p={p_value:.4f})"
            else:
                return f"Baseline significantly outperforms model (p={p_value:.4f})"
        else:
            return f"No significant difference (p={p_value:.4f})"
    
    def bootstrap_r2_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Compute bootstrap confidence interval for R².
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Model predictions
        n_bootstrap : int
            Number of bootstrap samples (default: 1000)
        confidence_level : float
            Confidence level (default: 0.95)
            
        Returns
        -------
        Dict containing:
            - r2: Original R² score
            - ci_lower: Lower bound of CI
            - ci_upper: Upper bound of CI
            - std_error: Standard error of R²
            - bootstrap_r2s: All bootstrap R² values
        """
        n = len(y_true)
        bootstrap_r2s = []
        
        np.random.seed(self.random_state)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Compute R² for bootstrap sample
            r2_boot = r2_score(y_true_boot, y_pred_boot)
            bootstrap_r2s.append(r2_boot)
        
        bootstrap_r2s = np.array(bootstrap_r2s)
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_r2s, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_r2s, 100 * (1 - alpha / 2))
        
        # Original R²
        r2_original = r2_score(y_true, y_pred)
        
        return {
            'r2': float(r2_original),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'std_error': float(np.std(bootstrap_r2s)),
            'bootstrap_r2s': bootstrap_r2s,
            'confidence_level': confidence_level,
            'interpretation': self._interpret_r2_ci(r2_original, ci_lower, ci_upper)
        }
    
    def _interpret_r2_ci(self, r2: float, ci_lower: float, ci_upper: float) -> str:
        """Interpret R² confidence interval."""
        if ci_upper < 0.01:
            return f"R²={r2:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] - No predictive power"
        elif ci_lower > 0:
            return f"R²={r2:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] - Significant positive predictive power"
        else:
            return f"R²={r2:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] - Uncertain predictive power"
    
    def f_test_model_comparison(
        self,
        y_true: np.ndarray,
        y_pred_model1: np.ndarray,
        y_pred_model2: np.ndarray,
        n_params_model1: int,
        n_params_model2: int,
        alpha: float = 0.05
    ) -> Dict:
        """
        F-test to compare two nested models.
        
        Tests if additional parameters in model2 significantly improve fit.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values
        y_pred_model1 : np.ndarray
            Predictions from simpler model
        y_pred_model2 : np.ndarray
            Predictions from complex model
        n_params_model1 : int
            Number of parameters in model 1
        n_params_model2 : int
            Number of parameters in model 2
        alpha : float
            Significance level
            
        Returns
        -------
        Dict with F-statistic, p-value, and interpretation
        """
        n = len(y_true)
        
        # RSS for both models
        rss1 = np.sum((y_true - y_pred_model1) ** 2)
        rss2 = np.sum((y_true - y_pred_model2) ** 2)
        
        # Degrees of freedom
        df1 = n_params_model2 - n_params_model1
        df2 = n - n_params_model2
        
        # F-statistic
        f_stat = ((rss1 - rss2) / df1) / (rss2 / df2) if df2 > 0 and rss2 > 0 else 0
        
        # P-value
        p_value = 1 - stats.f.cdf(f_stat, df1, df2) if f_stat > 0 else 1.0
        
        return {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'df1': df1,
            'df2': df2,
            'significant': p_value < alpha,
            'rss_model1': float(rss1),
            'rss_model2': float(rss2),
            'interpretation': self._interpret_f_test(p_value, alpha)
        }
    
    def _interpret_f_test(self, p_value: float, alpha: float) -> str:
        """Interpret F-test results."""
        if p_value < alpha:
            return f"Complex model significantly better (p={p_value:.4f})"
        else:
            return f"No significant improvement (p={p_value:.4f})"
    
    def multiple_testing_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni',
        alpha: float = 0.05
    ) -> Dict:
        """
        Apply multiple testing correction to p-values.
        
        Parameters
        ----------
        p_values : List[float]
            List of p-values from multiple tests
        method : str
            Correction method: 'bonferroni' or 'fdr' (Benjamini-Hochberg)
        alpha : float
            Family-wise error rate
            
        Returns
        -------
        Dict with corrected p-values and significance decisions
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            # Bonferroni correction
            corrected_alpha = alpha / n_tests
            significant = p_values < corrected_alpha
            corrected_p = np.minimum(p_values * n_tests, 1.0)
            
        elif method == 'fdr':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            # Find largest i where p(i) <= (i/n) * alpha
            threshold = np.arange(1, n_tests + 1) / n_tests * alpha
            significant_sorted = sorted_p <= threshold
            
            if np.any(significant_sorted):
                max_i = np.where(significant_sorted)[0][-1]
                significant = np.zeros(n_tests, dtype=bool)
                significant[sorted_indices[:max_i + 1]] = True
            else:
                significant = np.zeros(n_tests, dtype=bool)
            
            # Corrected p-values (q-values)
            corrected_p = np.minimum.accumulate(
                sorted_p * n_tests / np.arange(1, n_tests + 1)
            )[np.argsort(sorted_indices)]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'original_p_values': p_values.tolist(),
            'corrected_p_values': corrected_p.tolist(),
            'significant': significant.tolist(),
            'n_significant': int(np.sum(significant)),
            'method': method,
            'alpha': alpha,
            'corrected_alpha': alpha / n_tests if method == 'bonferroni' else None
        }
    
    def permutation_test_r2(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_permutations: int = 1000,
        alpha: float = 0.05
    ) -> Dict:
        """
        Permutation test to assess if R² is significantly different from zero.
        
        H₀: Features have no predictive power (R² = 0)
        H₁: Features have predictive power (R² > 0)
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Model predictions
        n_permutations : int
            Number of permutations
        alpha : float
            Significance level
            
        Returns
        -------
        Dict with test results
        """
        # Original R²
        r2_original = r2_score(y_true, y_pred)
        
        # Permutation distribution
        np.random.seed(self.random_state)
        r2_permuted = []
        
        for _ in range(n_permutations):
            # Shuffle target values
            y_true_perm = np.random.permutation(y_true)
            r2_perm = r2_score(y_true_perm, y_pred)
            r2_permuted.append(r2_perm)
        
        r2_permuted = np.array(r2_permuted)
        
        # P-value: proportion of permuted R² >= original R²
        p_value = np.mean(r2_permuted >= r2_original)
        
        return {
            'r2_original': float(r2_original),
            'r2_permuted_mean': float(np.mean(r2_permuted)),
            'r2_permuted_std': float(np.std(r2_permuted)),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'n_permutations': n_permutations,
            'interpretation': self._interpret_permutation_test(p_value, r2_original, alpha)
        }
    
    def _interpret_permutation_test(self, p_value: float, r2: float, alpha: float) -> str:
        """Interpret permutation test results."""
        if p_value < alpha:
            return f"R²={r2:.4f} is significantly different from zero (p={p_value:.4f})"
        else:
            return f"R²={r2:.4f} is not significantly different from zero (p={p_value:.4f})"
    
    def comprehensive_model_test(
        self,
        y_true: np.ndarray,
        y_pred_model: np.ndarray,
        y_pred_baseline: np.ndarray,
        model_name: str = "Model",
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
        alpha: float = 0.05
    ) -> Dict:
        """
        Run comprehensive statistical tests for a model.
        
        Includes:
        - Bootstrap CI for R²
        - T-test vs baseline
        - Permutation test
        - All metrics with CIs
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred_model : np.ndarray
            Model predictions
        y_pred_baseline : np.ndarray
            Baseline predictions
        model_name : str
            Name of the model
        n_bootstrap : int
            Number of bootstrap samples
        n_permutations : int
            Number of permutations
        alpha : float
            Significance level
            
        Returns
        -------
        Dict with all test results
        """
        self.logger.info(f"\nRunning comprehensive statistical tests for {model_name}...")
        
        results = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'alpha': alpha
        }
        
        # 1. Bootstrap CI for R²
        self.logger.info("  Computing bootstrap CI for R²...")
        results['r2_bootstrap'] = self.bootstrap_r2_ci(
            y_true, y_pred_model, n_bootstrap, confidence_level=1-alpha
        )
        
        # 2. T-test vs baseline
        self.logger.info("  Running t-test vs baseline...")
        results['t_test_vs_baseline'] = self.test_model_vs_baseline(
            y_true, y_pred_model, y_pred_baseline, alpha
        )
        
        # 3. Permutation test
        self.logger.info("  Running permutation test...")
        results['permutation_test'] = self.permutation_test_r2(
            y_true, y_pred_model, n_permutations, alpha
        )
        
        # 4. Basic metrics
        results['metrics'] = {
            'r2': float(r2_score(y_true, y_pred_model)),
            'mae': float(mean_absolute_error(y_true, y_pred_model)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred_model)))
        }
        
        self.logger.info("  ✓ Statistical tests complete")
        
        return results
    
    def format_results_table(self, results: Dict) -> pd.DataFrame:
        """
        Format comprehensive test results as a DataFrame.
        
        Parameters
        ----------
        results : Dict
            Results from comprehensive_model_test
            
        Returns
        -------
        pd.DataFrame with formatted results
        """
        data = {
            'Metric': [],
            'Value': [],
            'CI_Lower': [],
            'CI_Upper': [],
            'P_Value': [],
            'Significant': []
        }
        
        # R² with bootstrap CI
        r2_boot = results['r2_bootstrap']
        data['Metric'].append('R²')
        data['Value'].append(f"{r2_boot['r2']:.4f}")
        data['CI_Lower'].append(f"{r2_boot['ci_lower']:.4f}")
        data['CI_Upper'].append(f"{r2_boot['ci_upper']:.4f}")
        data['P_Value'].append(f"{results['permutation_test']['p_value']:.4f}")
        data['Significant'].append('Yes' if results['permutation_test']['significant'] else 'No')
        
        # MAE
        data['Metric'].append('MAE')
        data['Value'].append(f"{results['metrics']['mae']:.4f}")
        data['CI_Lower'].append('-')
        data['CI_Upper'].append('-')
        data['P_Value'].append('-')
        data['Significant'].append('-')
        
        # RMSE
        data['Metric'].append('RMSE')
        data['Value'].append(f"{results['metrics']['rmse']:.4f}")
        data['CI_Lower'].append('-')
        data['CI_Upper'].append('-')
        data['P_Value'].append('-')
        data['Significant'].append('-')
        
        # T-test vs baseline
        t_test = results['t_test_vs_baseline']
        data['Metric'].append('vs Baseline')
        data['Value'].append(f"t={t_test['t_statistic']:.2f}")
        data['CI_Lower'].append('-')
        data['CI_Upper'].append('-')
        data['P_Value'].append(f"{t_test['p_value']:.4f}")
        data['Significant'].append('Yes' if t_test['significant'] else 'No')
        
        return pd.DataFrame(data)
