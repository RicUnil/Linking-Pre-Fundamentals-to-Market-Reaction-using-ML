"""
Market Regime Analysis Module

This module provides market regime analysis:
- Identify bull vs bear markets
- Test if predictability varies by market regime
- Analyze model performance in different market conditions

Author: Academic Project Enhancement
Date: December 2025
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MarketRegimeAnalyzer:
    """
    Market regime analysis for testing EMH under different conditions.
    
    Tests whether market efficiency varies between bull and bear markets.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize market regime analyzer.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def identify_market_regimes(
        self,
        spy_returns: pd.Series,
        method: str = 'moving_average',
        window: int = 60
    ) -> pd.Series:
        """
        Identify bull and bear market regimes.
        
        Parameters
        ----------
        spy_returns : pd.Series
            SPY returns with datetime index
        method : str
            'moving_average' or 'drawdown'
        window : int
            Window for moving average (in days)
            
        Returns
        -------
        pd.Series with regime labels ('bull' or 'bear')
        """
        self.logger.info(f"Identifying market regimes using {method} method...")
        
        if method == 'moving_average':
            # Bull: price above MA, Bear: price below MA
            ma = spy_returns.rolling(window=window).mean()
            regime = pd.Series('bull', index=spy_returns.index)
            regime[spy_returns < ma] = 'bear'
            
        elif method == 'drawdown':
            # Bull: drawdown < threshold, Bear: drawdown > threshold
            cumulative = (1 + spy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            regime = pd.Series('bull', index=spy_returns.index)
            regime[drawdown < -0.10] = 'bear'  # 10% drawdown threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_bull = (regime == 'bull').sum()
        n_bear = (regime == 'bear').sum()
        
        self.logger.info(f"  Bull periods: {n_bull} ({n_bull/(n_bull+n_bear)*100:.1f}%)")
        self.logger.info(f"  Bear periods: {n_bear} ({n_bear/(n_bull+n_bear)*100:.1f}%)")
        
        return regime
    
    def map_regimes_to_events(
        self,
        earnings_dates: pd.Series,
        spy_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Map market regimes to earnings events.
        
        Parameters
        ----------
        earnings_dates : pd.Series
            Earnings announcement dates
        spy_data : pd.DataFrame
            SPY data with 'Date' and 'adj_close' columns
            
        Returns
        -------
        np.ndarray with regime labels for each event
        """
        self.logger.info("Mapping regimes to earnings events...")
        
        # Ensure dates are datetime
        earnings_dates = pd.to_datetime(earnings_dates)
        spy_data = spy_data.copy()
        
        if 'Date' in spy_data.columns:
            spy_data['Date'] = pd.to_datetime(spy_data['Date'])
            spy_data = spy_data.set_index('Date')
        
        # Compute SPY returns
        spy_returns = spy_data['adj_close'].pct_change()
        
        # Identify regimes
        regimes = self.identify_market_regimes(spy_returns)
        
        # Map to earnings dates
        event_regimes = []
        for date in earnings_dates:
            # Find closest date in SPY data
            if date in regimes.index:
                event_regimes.append(regimes.loc[date])
            else:
                # Find nearest date
                idx = regimes.index.get_indexer([date], method='nearest')[0]
                event_regimes.append(regimes.iloc[idx])
        
        event_regimes = np.array(event_regimes)
        
        n_bull = (event_regimes == 'bull').sum()
        n_bear = (event_regimes == 'bear').sum()
        
        self.logger.info(f"  Events in bull market: {n_bull} ({n_bull/(n_bull+n_bear)*100:.1f}%)")
        self.logger.info(f"  Events in bear market: {n_bear} ({n_bear/(n_bull+n_bear)*100:.1f}%)")
        
        return event_regimes
    
    def train_regime_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        regimes_train: np.ndarray,
        regimes_test: np.ndarray,
        min_samples: int = 100
    ) -> Dict:
        """
        Train separate models for each market regime.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        regimes_train : np.ndarray
            Regime labels for training
        regimes_test : np.ndarray
            Regime labels for test
        min_samples : int
            Minimum samples per regime
            
        Returns
        -------
        Dict with regime-specific results
        """
        self.logger.info("Training regime-specific models...")
        
        results = {}
        
        for regime in ['bull', 'bear']:
            train_mask = regimes_train == regime
            test_mask = regimes_test == regime
            
            n_train = np.sum(train_mask)
            n_test = np.sum(test_mask)
            
            if n_train < min_samples or n_test < 50:
                self.logger.info(f"  Skipping {regime}: insufficient samples (train={n_train}, test={n_test})")
                continue
            
            self.logger.info(f"\n  Training models for {regime} market (train={n_train}, test={n_test})")
            
            X_train_regime = X_train[train_mask]
            y_train_regime = y_train[train_mask]
            X_test_regime = X_test[test_mask]
            y_test_regime = y_test[test_mask]
            
            regime_results = {
                'n_train': int(n_train),
                'n_test': int(n_test),
                'models': {}
            }
            
            # Train baseline
            baseline = DummyRegressor(strategy='mean')
            baseline.fit(X_train_regime, y_train_regime)
            y_pred_baseline = baseline.predict(X_test_regime)
            
            regime_results['models']['baseline'] = {
                'r2': float(r2_score(y_test_regime, y_pred_baseline)),
                'mae': float(mean_absolute_error(y_test_regime, y_pred_baseline)),
                'rmse': float(np.sqrt(mean_squared_error(y_test_regime, y_pred_baseline)))
            }
            
            # Train Ridge
            ridge = Ridge(alpha=1.0, random_state=self.random_state)
            ridge.fit(X_train_regime, y_train_regime)
            y_pred_ridge = ridge.predict(X_test_regime)
            
            regime_results['models']['ridge'] = {
                'r2': float(r2_score(y_test_regime, y_pred_ridge)),
                'mae': float(mean_absolute_error(y_test_regime, y_pred_ridge)),
                'rmse': float(np.sqrt(mean_squared_error(y_test_regime, y_pred_ridge)))
            }
            
            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                min_samples_split=20,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_train_regime, y_train_regime)
            y_pred_rf = rf.predict(X_test_regime)
            
            regime_results['models']['random_forest'] = {
                'r2': float(r2_score(y_test_regime, y_pred_rf)),
                'mae': float(mean_absolute_error(y_test_regime, y_pred_rf)),
                'rmse': float(np.sqrt(mean_squared_error(y_test_regime, y_pred_rf)))
            }
            
            # Best model
            best_r2 = max(
                regime_results['models']['baseline']['r2'],
                regime_results['models']['ridge']['r2'],
                regime_results['models']['random_forest']['r2']
            )
            regime_results['best_r2'] = float(best_r2)
            
            self.logger.info(f"    Best R² for {regime} market: {best_r2:.4f}")
            
            results[regime] = regime_results
        
        self.logger.info(f"\n  ✓ Trained models for {len(results)} regimes")
        
        return results
    
    def compare_regimes(
        self,
        regime_results: Dict
    ) -> pd.DataFrame:
        """
        Compare model performance across market regimes.
        
        Parameters
        ----------
        regime_results : Dict
            Results from train_regime_models
            
        Returns
        -------
        pd.DataFrame with regime comparison
        """
        self.logger.info("Comparing regime performance...")
        
        comparison_data = []
        
        for regime, results in regime_results.items():
            for model_name, metrics in results['models'].items():
                comparison_data.append({
                    'regime': regime,
                    'model': model_name,
                    'n_train': results['n_train'],
                    'n_test': results['n_test'],
                    'r2': metrics['r2'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse']
                })
        
        df = pd.DataFrame(comparison_data)
        
        return df
    
    def test_regime_difference(
        self,
        regime_results: Dict,
        alpha: float = 0.05
    ) -> Dict:
        """
        Test if there's significant difference between bull and bear markets.
        
        Parameters
        ----------
        regime_results : Dict
            Results from train_regime_models
        alpha : float
            Significance level
            
        Returns
        -------
        Dict with test results
        """
        self.logger.info("Testing regime differences...")
        
        if 'bull' not in regime_results or 'bear' not in regime_results:
            return {'error': 'Both bull and bear regimes required'}
        
        bull_r2 = regime_results['bull']['best_r2']
        bear_r2 = regime_results['bear']['best_r2']
        
        diff = bull_r2 - bear_r2
        
        # Simple comparison (could add statistical test)
        interpretation = ""
        if abs(diff) < 0.01:
            interpretation = "No meaningful difference between regimes"
        elif bull_r2 > bear_r2:
            interpretation = f"Bull market slightly more predictable (Δ={diff:.4f})"
        else:
            interpretation = f"Bear market slightly more predictable (Δ={diff:.4f})"
        
        self.logger.info(f"  {interpretation}")
        
        return {
            'bull_r2': float(bull_r2),
            'bear_r2': float(bear_r2),
            'difference': float(diff),
            'interpretation': interpretation
        }
    
    def comprehensive_regime_analysis(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        earnings_dates_train: pd.Series,
        earnings_dates_test: pd.Series,
        spy_data: pd.DataFrame,
        min_samples: int = 100
    ) -> Dict:
        """
        Run comprehensive market regime analysis.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        earnings_dates_train : pd.Series
            Earnings dates for training
        earnings_dates_test : pd.Series
            Earnings dates for test
        spy_data : pd.DataFrame
            SPY price data
        min_samples : int
            Minimum samples per regime
            
        Returns
        -------
        Dict with all regime analysis results
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("COMPREHENSIVE MARKET REGIME ANALYSIS")
        self.logger.info("=" * 70)
        
        results = {}
        
        # 1. Map regimes to events
        self.logger.info("\n1. Mapping Market Regimes to Events")
        regimes_train = self.map_regimes_to_events(earnings_dates_train, spy_data)
        regimes_test = self.map_regimes_to_events(earnings_dates_test, spy_data)
        
        results['regimes_train'] = regimes_train
        results['regimes_test'] = regimes_test
        
        # 2. Train regime-specific models
        self.logger.info("\n2. Training Regime-Specific Models")
        results['regime_models'] = self.train_regime_models(
            X_train, y_train, X_test, y_test,
            regimes_train, regimes_test, min_samples
        )
        
        # 3. Compare regimes
        self.logger.info("\n3. Comparing Regimes")
        results['regime_comparison'] = self.compare_regimes(results['regime_models'])
        
        # 4. Test differences
        self.logger.info("\n4. Testing Regime Differences")
        results['regime_difference'] = self.test_regime_difference(results['regime_models'])
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("MARKET REGIME ANALYSIS COMPLETE")
        self.logger.info("=" * 70)
        
        return results
    
    def create_regime_summary(self, results: Dict) -> str:
        """
        Create text summary of regime analysis.
        
        Parameters
        ----------
        results : Dict
            Results from comprehensive_regime_analysis
            
        Returns
        -------
        str with formatted summary
        """
        lines = []
        lines.append("=" * 70)
        lines.append("MARKET REGIME ANALYSIS SUMMARY")
        lines.append("=" * 70)
        
        # Regime distribution
        regimes_test = results['regimes_test']
        n_bull = (regimes_test == 'bull').sum()
        n_bear = (regimes_test == 'bear').sum()
        
        lines.append(f"\nTest set regime distribution:")
        lines.append(f"  Bull market: {n_bull} events ({n_bull/(n_bull+n_bear)*100:.1f}%)")
        lines.append(f"  Bear market: {n_bear} events ({n_bear/(n_bull+n_bear)*100:.1f}%)")
        
        # Performance comparison
        diff = results['regime_difference']
        lines.append(f"\nPredictability comparison:")
        lines.append(f"  Bull market R²: {diff['bull_r2']:.4f}")
        lines.append(f"  Bear market R²: {diff['bear_r2']:.4f}")
        lines.append(f"  {diff['interpretation']}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
