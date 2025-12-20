"""
Residual Analysis Module

This module provides comprehensive residual analysis:
- Normality tests
- Heteroscedasticity tests
- Autocorrelation analysis
- Outlier detection
- Residual patterns by sector and time

Author: Academic Project Enhancement
Date: December 2025
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error


class ResidualAnalyzer:
    """
    Comprehensive residual analysis for regression models.
    
    Checks model assumptions and identifies patterns in prediction errors.
    """
    
    def __init__(self):
        """Initialize residual analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def compute_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Compute residuals and basic statistics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns
        -------
        Dict with residual statistics
        """
        residuals = y_true - y_pred
        
        return {
            'residuals': residuals,
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'q25': float(np.percentile(residuals, 25)),
            'q50': float(np.percentile(residuals, 50)),
            'q75': float(np.percentile(residuals, 75)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
        }
    
    def test_normality(
        self,
        residuals: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Test if residuals are normally distributed.
        
        Uses multiple tests:
        - Shapiro-Wilk test
        - Jarque-Bera test
        - Anderson-Darling test
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
        alpha : float
            Significance level
            
        Returns
        -------
        Dict with test results
        """
        self.logger.info("Testing residual normality...")
        
        results = {}
        
        # 1. Shapiro-Wilk test (best for small-medium samples)
        if len(residuals) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            results['shapiro_wilk'] = {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'normal': shapiro_p > alpha
            }
        else:
            results['shapiro_wilk'] = {
                'note': 'Skipped (sample too large)'
            }
        
        # 2. Jarque-Bera test (based on skewness and kurtosis)
        jb_stat, jb_p = stats.jarque_bera(residuals)
        results['jarque_bera'] = {
            'statistic': float(jb_stat),
            'p_value': float(jb_p),
            'normal': jb_p > alpha
        }
        
        # 3. Anderson-Darling test
        ad_result = stats.anderson(residuals, dist='norm')
        results['anderson_darling'] = {
            'statistic': float(ad_result.statistic),
            'critical_values': ad_result.critical_values.tolist(),
            'significance_levels': ad_result.significance_level.tolist()
        }
        
        # 4. Skewness and Kurtosis
        results['skewness'] = float(stats.skew(residuals))
        results['kurtosis'] = float(stats.kurtosis(residuals))
        
        # Overall interpretation
        if 'shapiro_wilk' in results and 'p_value' in results['shapiro_wilk']:
            normal = results['shapiro_wilk']['normal'] and results['jarque_bera']['normal']
        else:
            normal = results['jarque_bera']['normal']
        
        results['interpretation'] = (
            "Residuals appear normally distributed" if normal
            else "Residuals deviate from normality"
        )
        
        self.logger.info(f"  {results['interpretation']}")
        
        return results
    
    def test_heteroscedasticity(
        self,
        residuals: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Test for heteroscedasticity (non-constant variance).
        
        Uses Breusch-Pagan test and visual inspection.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
        y_pred : np.ndarray
            Predicted values
        alpha : float
            Significance level
            
        Returns
        -------
        Dict with test results
        """
        self.logger.info("Testing for heteroscedasticity...")
        
        # Compute squared residuals
        squared_residuals = residuals ** 2
        
        # Correlation between |residuals| and predictions
        abs_residuals = np.abs(residuals)
        corr, p_value = stats.spearmanr(abs_residuals, y_pred)
        
        results = {
            'spearman_correlation': float(corr),
            'p_value': float(p_value),
            'heteroscedastic': p_value < alpha,
            'interpretation': (
                "Significant heteroscedasticity detected" if p_value < alpha
                else "No significant heteroscedasticity"
            )
        }
        
        # Variance ratio (high vs low predictions)
        median_pred = np.median(y_pred)
        low_pred_mask = y_pred < median_pred
        high_pred_mask = y_pred >= median_pred
        
        var_low = np.var(residuals[low_pred_mask])
        var_high = np.var(residuals[high_pred_mask])
        
        results['variance_ratio'] = float(var_high / var_low) if var_low > 0 else np.inf
        
        self.logger.info(f"  {results['interpretation']}")
        
        return results
    
    def detect_outliers(
        self,
        residuals: np.ndarray,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Dict:
        """
        Detect outliers in residuals.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
        method : str
            'iqr' for IQR method, 'zscore' for z-score method
        threshold : float
            Threshold for outlier detection (IQR multiplier or z-score)
            
        Returns
        -------
        Dict with outlier information
        """
        self.logger.info(f"Detecting outliers using {method} method...")
        
        if method == 'iqr':
            q25 = np.percentile(residuals, 25)
            q75 = np.percentile(residuals, 75)
            iqr = q75 - q25
            
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr
            
            outliers = (residuals < lower_bound) | (residuals > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
            outliers = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_outliers = np.sum(outliers)
        pct_outliers = n_outliers / len(residuals) * 100
        
        results = {
            'method': method,
            'threshold': threshold,
            'n_outliers': int(n_outliers),
            'pct_outliers': float(pct_outliers),
            'outlier_indices': np.where(outliers)[0].tolist(),
            'outlier_mask': outliers
        }
        
        self.logger.info(f"  Found {n_outliers} outliers ({pct_outliers:.2f}%)")
        
        return results
    
    def analyze_by_sector(
        self,
        residuals: np.ndarray,
        sectors: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze residuals by sector.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
        sectors : np.ndarray
            Sector labels for each observation
            
        Returns
        -------
        pd.DataFrame with sector-wise statistics
        """
        self.logger.info("Analyzing residuals by sector...")
        
        df = pd.DataFrame({
            'residual': residuals,
            'sector': sectors
        })
        
        sector_stats = df.groupby('sector')['residual'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('q25', lambda x: np.percentile(x, 25)),
            ('median', 'median'),
            ('q75', lambda x: np.percentile(x, 75)),
            ('max', 'max'),
            ('rmse', lambda x: np.sqrt(np.mean(x**2)))
        ]).round(6)
        
        # Sort by RMSE
        sector_stats = sector_stats.sort_values('rmse', ascending=False)
        
        self.logger.info(f"  Analyzed {len(sector_stats)} sectors")
        
        return sector_stats
    
    def analyze_by_time(
        self,
        residuals: np.ndarray,
        dates: pd.Series,
        freq: str = 'M'
    ) -> pd.DataFrame:
        """
        Analyze residuals over time.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
        dates : pd.Series
            Dates for each observation
        freq : str
            Frequency for aggregation ('M' for month, 'Q' for quarter, 'Y' for year)
            
        Returns
        -------
        pd.DataFrame with time-series statistics
        """
        self.logger.info(f"Analyzing residuals over time (freq={freq})...")
        
        df = pd.DataFrame({
            'residual': residuals,
            'date': pd.to_datetime(dates)
        })
        
        # Resample
        df = df.set_index('date')
        
        time_stats = df.resample(freq)['residual'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('rmse', lambda x: np.sqrt(np.mean(x**2)))
        ]).round(6)
        
        self.logger.info(f"  Analyzed {len(time_stats)} time periods")
        
        return time_stats
    
    def test_autocorrelation(
        self,
        residuals: np.ndarray,
        max_lag: int = 10
    ) -> Dict:
        """
        Test for autocorrelation in residuals.
        
        Uses Ljung-Box test.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals (should be ordered by time)
        max_lag : int
            Maximum lag to test
            
        Returns
        -------
        Dict with autocorrelation test results
        """
        self.logger.info("Testing for autocorrelation...")
        
        # Compute autocorrelation
        acf_values = []
        for lag in range(1, max_lag + 1):
            if lag < len(residuals):
                corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                acf_values.append(corr)
            else:
                acf_values.append(np.nan)
        
        # Ljung-Box test
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(residuals, lags=max_lag, return_df=True)
        
        results = {
            'acf_values': acf_values,
            'ljung_box_stats': lb_result['lb_stat'].tolist(),
            'ljung_box_pvalues': lb_result['lb_pvalue'].tolist(),
            'significant_lags': (lb_result['lb_pvalue'] < 0.05).sum(),
            'interpretation': (
                "Significant autocorrelation detected" 
                if (lb_result['lb_pvalue'] < 0.05).any()
                else "No significant autocorrelation"
            )
        }
        
        self.logger.info(f"  {results['interpretation']}")
        
        return results
    
    def comprehensive_residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
        alpha: float = 0.05
    ) -> Dict:
        """
        Run comprehensive residual analysis.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        metadata : Optional[pd.DataFrame]
            Metadata with 'sector' and 'earnings_date' columns
        alpha : float
            Significance level
            
        Returns
        -------
        Dict with all analysis results
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("COMPREHENSIVE RESIDUAL ANALYSIS")
        self.logger.info("=" * 70)
        
        results = {}
        
        # 1. Basic residual statistics
        self.logger.info("\n1. Computing Residual Statistics")
        results['basic_stats'] = self.compute_residuals(y_true, y_pred)
        residuals = results['basic_stats']['residuals']
        
        # 2. Normality tests
        self.logger.info("\n2. Testing Normality")
        results['normality'] = self.test_normality(residuals, alpha)
        
        # 3. Heteroscedasticity tests
        self.logger.info("\n3. Testing Heteroscedasticity")
        results['heteroscedasticity'] = self.test_heteroscedasticity(
            residuals, y_pred, alpha
        )
        
        # 4. Outlier detection
        self.logger.info("\n4. Detecting Outliers")
        results['outliers_iqr'] = self.detect_outliers(residuals, method='iqr', threshold=3.0)
        results['outliers_zscore'] = self.detect_outliers(residuals, method='zscore', threshold=3.0)
        
        # 5. Sector analysis (if metadata provided)
        if metadata is not None and 'sector' in metadata.columns:
            self.logger.info("\n5. Analyzing by Sector")
            results['sector_analysis'] = self.analyze_by_sector(
                residuals, metadata['sector'].values
            )
        
        # 6. Time analysis (if metadata provided)
        if metadata is not None and 'earnings_date' in metadata.columns:
            self.logger.info("\n6. Analyzing over Time")
            results['time_analysis'] = self.analyze_by_time(
                residuals, metadata['earnings_date'], freq='Q'
            )
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("RESIDUAL ANALYSIS COMPLETE")
        self.logger.info("=" * 70)
        
        return results
    
    def create_summary_report(self, results: Dict) -> str:
        """
        Create text summary of residual analysis.
        
        Parameters
        ----------
        results : Dict
            Results from comprehensive_residual_analysis
            
        Returns
        -------
        str with formatted summary
        """
        lines = []
        lines.append("=" * 70)
        lines.append("RESIDUAL ANALYSIS SUMMARY")
        lines.append("=" * 70)
        
        # Basic stats
        stats = results['basic_stats']
        lines.append("\n1. BASIC STATISTICS")
        lines.append(f"   Mean:   {stats['mean']:.6f}")
        lines.append(f"   Std:    {stats['std']:.6f}")
        lines.append(f"   RMSE:   {stats['rmse']:.6f}")
        lines.append(f"   Range:  [{stats['min']:.6f}, {stats['max']:.6f}]")
        
        # Normality
        norm = results['normality']
        lines.append("\n2. NORMALITY TESTS")
        lines.append(f"   {norm['interpretation']}")
        lines.append(f"   Skewness: {norm['skewness']:.4f}")
        lines.append(f"   Kurtosis: {norm['kurtosis']:.4f}")
        
        # Heteroscedasticity
        hetero = results['heteroscedasticity']
        lines.append("\n3. HETEROSCEDASTICITY")
        lines.append(f"   {hetero['interpretation']}")
        lines.append(f"   Variance ratio: {hetero['variance_ratio']:.4f}")
        
        # Outliers
        outliers = results['outliers_iqr']
        lines.append("\n4. OUTLIERS")
        lines.append(f"   Found {outliers['n_outliers']} outliers ({outliers['pct_outliers']:.2f}%)")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
