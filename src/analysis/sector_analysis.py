"""
Sector-Specific Analysis Module

This module provides sector-specific model analysis:
- Train separate models per sector
- Compare performance across sectors
- Identify sector-specific patterns
- Test if certain sectors are more/less predictable

Author: Academic Project Enhancement
Date: December 2025
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SectorAnalyzer:
    """
    Sector-specific model analysis.
    
    Tests whether market efficiency varies by sector.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize sector analyzer.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def train_sector_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sectors_train: np.ndarray,
        sectors_test: np.ndarray,
        min_samples: int = 50
    ) -> Dict:
        """
        Train separate models for each sector.
        
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
        sectors_train : np.ndarray
            Sector labels for training data
        sectors_test : np.ndarray
            Sector labels for test data
        min_samples : int
            Minimum samples required to train sector model
            
        Returns
        -------
        Dict with sector-specific results
        """
        self.logger.info("Training sector-specific models...")
        
        unique_sectors = np.unique(sectors_train)
        results = {}
        
        for sector in unique_sectors:
            # Get sector data
            train_mask = sectors_train == sector
            test_mask = sectors_test == sector
            
            n_train = np.sum(train_mask)
            n_test = np.sum(test_mask)
            
            if n_train < min_samples or n_test < 10:
                self.logger.info(f"  Skipping {sector}: insufficient samples (train={n_train}, test={n_test})")
                continue
            
            self.logger.info(f"\n  Training models for {sector} (train={n_train}, test={n_test})")
            
            X_train_sector = X_train[train_mask]
            y_train_sector = y_train[train_mask]
            X_test_sector = X_test[test_mask]
            y_test_sector = y_test[test_mask]
            
            sector_results = {
                'n_train': int(n_train),
                'n_test': int(n_test),
                'models': {}
            }
            
            # Train baseline
            baseline = DummyRegressor(strategy='mean')
            baseline.fit(X_train_sector, y_train_sector)
            y_pred_baseline = baseline.predict(X_test_sector)
            
            sector_results['models']['baseline'] = {
                'r2': float(r2_score(y_test_sector, y_pred_baseline)),
                'mae': float(mean_absolute_error(y_test_sector, y_pred_baseline)),
                'rmse': float(np.sqrt(mean_squared_error(y_test_sector, y_pred_baseline)))
            }
            
            # Train Ridge
            ridge = Ridge(alpha=1.0, random_state=self.random_state)
            ridge.fit(X_train_sector, y_train_sector)
            y_pred_ridge = ridge.predict(X_test_sector)
            
            sector_results['models']['ridge'] = {
                'r2': float(r2_score(y_test_sector, y_pred_ridge)),
                'mae': float(mean_absolute_error(y_test_sector, y_pred_ridge)),
                'rmse': float(np.sqrt(mean_squared_error(y_test_sector, y_pred_ridge)))
            }
            
            # Train Random Forest (smaller for speed)
            rf = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                min_samples_split=20,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_train_sector, y_train_sector)
            y_pred_rf = rf.predict(X_test_sector)
            
            sector_results['models']['random_forest'] = {
                'r2': float(r2_score(y_test_sector, y_pred_rf)),
                'mae': float(mean_absolute_error(y_test_sector, y_pred_rf)),
                'rmse': float(np.sqrt(mean_squared_error(y_test_sector, y_pred_rf)))
            }
            
            # Best model
            best_r2 = max(
                sector_results['models']['baseline']['r2'],
                sector_results['models']['ridge']['r2'],
                sector_results['models']['random_forest']['r2']
            )
            sector_results['best_r2'] = float(best_r2)
            
            self.logger.info(f"    Best R² for {sector}: {best_r2:.4f}")
            
            results[sector] = sector_results
        
        self.logger.info(f"\n  ✓ Trained models for {len(results)} sectors")
        
        return results
    
    def compare_sectors(
        self,
        sector_results: Dict
    ) -> pd.DataFrame:
        """
        Compare model performance across sectors.
        
        Parameters
        ----------
        sector_results : Dict
            Results from train_sector_models
            
        Returns
        -------
        pd.DataFrame with sector comparison
        """
        self.logger.info("Comparing sector performance...")
        
        comparison_data = []
        
        for sector, results in sector_results.items():
            for model_name, metrics in results['models'].items():
                comparison_data.append({
                    'sector': sector,
                    'model': model_name,
                    'n_train': results['n_train'],
                    'n_test': results['n_test'],
                    'r2': metrics['r2'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse']
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by R² (descending)
        df = df.sort_values(['model', 'r2'], ascending=[True, False])
        
        return df
    
    def identify_predictable_sectors(
        self,
        sector_results: Dict,
        r2_threshold: float = 0.05
    ) -> Dict:
        """
        Identify sectors where models show some predictability.
        
        Parameters
        ----------
        sector_results : Dict
            Results from train_sector_models
        r2_threshold : float
            Threshold for considering a sector "predictable"
            
        Returns
        -------
        Dict with predictable and unpredictable sectors
        """
        self.logger.info(f"Identifying sectors with R² > {r2_threshold}...")
        
        predictable = []
        unpredictable = []
        
        for sector, results in sector_results.items():
            best_r2 = results['best_r2']
            
            if best_r2 > r2_threshold:
                predictable.append({
                    'sector': sector,
                    'best_r2': best_r2,
                    'n_test': results['n_test']
                })
            else:
                unpredictable.append({
                    'sector': sector,
                    'best_r2': best_r2,
                    'n_test': results['n_test']
                })
        
        # Sort by R²
        predictable = sorted(predictable, key=lambda x: x['best_r2'], reverse=True)
        unpredictable = sorted(unpredictable, key=lambda x: x['best_r2'], reverse=True)
        
        self.logger.info(f"  Predictable sectors: {len(predictable)}")
        self.logger.info(f"  Unpredictable sectors: {len(unpredictable)}")
        
        return {
            'predictable': predictable,
            'unpredictable': unpredictable,
            'n_predictable': len(predictable),
            'n_unpredictable': len(unpredictable),
            'pct_predictable': len(predictable) / (len(predictable) + len(unpredictable)) * 100
        }
    
    def test_sector_heterogeneity(
        self,
        sector_results: Dict
    ) -> Dict:
        """
        Test if there's significant heterogeneity across sectors.
        
        Uses ANOVA-like test on R² values.
        
        Parameters
        ----------
        sector_results : Dict
            Results from train_sector_models
            
        Returns
        -------
        Dict with heterogeneity test results
        """
        self.logger.info("Testing sector heterogeneity...")
        
        # Extract R² values for each model type
        r2_by_model = {
            'baseline': [],
            'ridge': [],
            'random_forest': []
        }
        
        for sector, results in sector_results.items():
            for model_name in r2_by_model.keys():
                r2_by_model[model_name].append(results['models'][model_name]['r2'])
        
        # Compute statistics
        heterogeneity = {}
        
        for model_name, r2_values in r2_by_model.items():
            r2_values = np.array(r2_values)
            
            heterogeneity[model_name] = {
                'mean_r2': float(np.mean(r2_values)),
                'std_r2': float(np.std(r2_values)),
                'min_r2': float(np.min(r2_values)),
                'max_r2': float(np.max(r2_values)),
                'range_r2': float(np.max(r2_values) - np.min(r2_values)),
                'cv_r2': float(np.std(r2_values) / np.abs(np.mean(r2_values))) if np.mean(r2_values) != 0 else np.inf
            }
        
        # Overall interpretation
        max_range = max(h['range_r2'] for h in heterogeneity.values())
        
        interpretation = (
            "Significant sector heterogeneity detected" if max_range > 0.1
            else "Limited sector heterogeneity"
        )
        
        self.logger.info(f"  {interpretation}")
        
        return {
            'by_model': heterogeneity,
            'max_range': float(max_range),
            'interpretation': interpretation
        }
    
    def comprehensive_sector_analysis(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sectors_train: np.ndarray,
        sectors_test: np.ndarray,
        min_samples: int = 50
    ) -> Dict:
        """
        Run comprehensive sector-specific analysis.
        
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
        sectors_train : np.ndarray
            Sector labels for training
        sectors_test : np.ndarray
            Sector labels for test
        min_samples : int
            Minimum samples per sector
            
        Returns
        -------
        Dict with all sector analysis results
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("COMPREHENSIVE SECTOR ANALYSIS")
        self.logger.info("=" * 70)
        
        results = {}
        
        # 1. Train sector-specific models
        self.logger.info("\n1. Training Sector-Specific Models")
        results['sector_models'] = self.train_sector_models(
            X_train, y_train, X_test, y_test,
            sectors_train, sectors_test, min_samples
        )
        
        # 2. Compare sectors
        self.logger.info("\n2. Comparing Sectors")
        results['sector_comparison'] = self.compare_sectors(results['sector_models'])
        
        # 3. Identify predictable sectors
        self.logger.info("\n3. Identifying Predictable Sectors")
        results['predictability'] = self.identify_predictable_sectors(
            results['sector_models'], r2_threshold=0.05
        )
        
        # 4. Test heterogeneity
        self.logger.info("\n4. Testing Sector Heterogeneity")
        results['heterogeneity'] = self.test_sector_heterogeneity(results['sector_models'])
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("SECTOR ANALYSIS COMPLETE")
        self.logger.info("=" * 70)
        
        return results
    
    def create_sector_summary(self, results: Dict) -> str:
        """
        Create text summary of sector analysis.
        
        Parameters
        ----------
        results : Dict
            Results from comprehensive_sector_analysis
            
        Returns
        -------
        str with formatted summary
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SECTOR ANALYSIS SUMMARY")
        lines.append("=" * 70)
        
        # Number of sectors
        n_sectors = len(results['sector_models'])
        lines.append(f"\nTotal sectors analyzed: {n_sectors}")
        
        # Predictability
        pred = results['predictability']
        lines.append(f"\nPredictable sectors (R² > 0.05): {pred['n_predictable']} ({pred['pct_predictable']:.1f}%)")
        lines.append(f"Unpredictable sectors: {pred['n_unpredictable']}")
        
        if pred['predictable']:
            lines.append("\nMost predictable sectors:")
            for item in pred['predictable'][:5]:
                lines.append(f"  - {item['sector']}: R² = {item['best_r2']:.4f}")
        
        # Heterogeneity
        hetero = results['heterogeneity']
        lines.append(f"\n{hetero['interpretation']}")
        lines.append(f"R² range across sectors: {hetero['max_range']:.4f}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
