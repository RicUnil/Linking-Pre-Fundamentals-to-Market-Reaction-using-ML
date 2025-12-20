"""
Feature Analysis Module

This module provides comprehensive feature importance and interpretability analysis:
- Feature importance from tree-based models
- SHAP values for model interpretability
- Permutation importance
- Feature correlation analysis

Author: Academic Project Enhancement
Date: December 2025
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge


class FeatureAnalyzer:
    """
    Comprehensive feature importance and interpretability analysis.
    
    Provides multiple methods to understand which features drive predictions
    and how they interact with the model.
    """
    
    def __init__(self, feature_names: List[str], random_state: int = 42):
        """
        Initialize feature analyzer.
        
        Parameters
        ----------
        feature_names : List[str]
            Names of features in the dataset
        random_state : int
            Random seed for reproducibility
        """
        self.feature_names = feature_names
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def compute_tree_importance(
        self,
        model: RandomForestRegressor,
        method: str = 'gain'
    ) -> pd.DataFrame:
        """
        Compute feature importance from tree-based model.
        
        Parameters
        ----------
        model : RandomForestRegressor
            Trained random forest model
        method : str
            'gain' for impurity-based or 'split' for split-based
            
        Returns
        -------
        pd.DataFrame with feature importances sorted by importance
        """
        self.logger.info("Computing tree-based feature importance...")
        
        if method == 'gain':
            importances = model.feature_importances_
        else:
            # Count splits (not directly available, use gain as proxy)
            importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Add percentage
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        # Add cumulative percentage
        importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
        
        self.logger.info(f"  Top 5 features: {importance_df.head()['feature'].tolist()}")
        
        return importance_df
    
    def compute_permutation_importance(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        scoring: str = 'r2'
    ) -> pd.DataFrame:
        """
        Compute permutation importance.
        
        More reliable than tree-based importance as it measures
        actual impact on model performance.
        
        Parameters
        ----------
        model : sklearn model
            Trained model
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        n_repeats : int
            Number of times to permute each feature
        scoring : str
            Scoring metric ('r2', 'neg_mean_squared_error', etc.)
            
        Returns
        -------
        pd.DataFrame with permutation importances
        """
        self.logger.info("Computing permutation importance...")
        
        # Compute permutation importance
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        # Add confidence intervals (±2 std)
        importance_df['ci_lower'] = (
            importance_df['importance_mean'] - 2 * importance_df['importance_std']
        )
        importance_df['ci_upper'] = (
            importance_df['importance_mean'] + 2 * importance_df['importance_std']
        )
        
        self.logger.info(f"  Top 5 features: {importance_df.head()['feature'].tolist()}")
        
        return importance_df
    
    def compute_shap_values(
        self,
        model,
        X: np.ndarray,
        model_type: str = 'tree',
        max_samples: int = 1000
    ) -> Dict:
        """
        Compute SHAP values for model interpretability.
        
        SHAP (SHapley Additive exPlanations) provides unified measure
        of feature importance based on game theory.
        
        Parameters
        ----------
        model : sklearn model
            Trained model
        X : np.ndarray
            Feature matrix
        model_type : str
            'tree' for tree-based models, 'linear' for linear models
        max_samples : int
            Maximum samples to use (for computational efficiency)
            
        Returns
        -------
        Dict containing:
            - shap_values: SHAP values array
            - feature_importance: Mean absolute SHAP values per feature
            - explainer: SHAP explainer object
        """
        self.logger.info("Computing SHAP values...")
        
        # Subsample if needed
        if len(X) > max_samples:
            self.logger.info(f"  Subsampling to {max_samples} samples for efficiency")
            np.random.seed(self.random_state)
            indices = np.random.choice(len(X), size=max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create appropriate explainer
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            # Use KernelExplainer as fallback (slower)
            explainer = shap.KernelExplainer(model.predict, X_sample)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Compute mean absolute SHAP values (feature importance)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)
        
        # Add percentage
        importance_df['importance_pct'] = (
            importance_df['shap_importance'] / importance_df['shap_importance'].sum() * 100
        )
        
        self.logger.info(f"  Top 5 features: {importance_df.head()['feature'].tolist()}")
        
        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'explainer': explainer,
            'X_sample': X_sample
        }
    
    def compute_linear_coefficients(
        self,
        model: Ridge,
        feature_std: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Extract and interpret linear model coefficients.
        
        Parameters
        ----------
        model : Ridge
            Trained Ridge regression model
        feature_std : Optional[np.ndarray]
            Standard deviations of features (for standardized coefficients)
            
        Returns
        -------
        pd.DataFrame with coefficients and interpretations
        """
        self.logger.info("Extracting linear model coefficients...")
        
        coef = model.coef_
        
        # Create DataFrame
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        })
        
        # If feature_std provided, compute standardized coefficients
        if feature_std is not None:
            # Standardized coefficient = coef * std(X)
            coef_df['std_coefficient'] = coef * feature_std
            coef_df['abs_std_coefficient'] = np.abs(coef_df['std_coefficient'])
            coef_df = coef_df.sort_values('abs_std_coefficient', ascending=False)
        else:
            coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        # Add interpretation
        coef_df['direction'] = coef_df['coefficient'].apply(
            lambda x: 'Positive' if x > 0 else 'Negative'
        )
        
        self.logger.info(f"  Top 5 features: {coef_df.head()['feature'].tolist()}")
        
        return coef_df
    
    def compare_importance_methods(
        self,
        tree_importance: pd.DataFrame,
        perm_importance: pd.DataFrame,
        shap_importance: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare feature importance across different methods.
        
        Parameters
        ----------
        tree_importance : pd.DataFrame
            Tree-based importance
        perm_importance : pd.DataFrame
            Permutation importance
        shap_importance : pd.DataFrame
            SHAP importance
            
        Returns
        -------
        pd.DataFrame with rankings from each method
        """
        self.logger.info("Comparing importance methods...")
        
        # Get rankings
        tree_rank = tree_importance.reset_index(drop=True).reset_index()
        tree_rank = tree_rank[['feature', 'index']].rename(columns={'index': 'tree_rank'})
        
        perm_rank = perm_importance.reset_index(drop=True).reset_index()
        perm_rank = perm_rank[['feature', 'index']].rename(columns={'index': 'perm_rank'})
        
        shap_rank = shap_importance.reset_index(drop=True).reset_index()
        shap_rank = shap_rank[['feature', 'index']].rename(columns={'index': 'shap_rank'})
        
        # Merge
        comparison = tree_rank.merge(perm_rank, on='feature').merge(shap_rank, on='feature')
        
        # Compute average rank
        comparison['avg_rank'] = comparison[['tree_rank', 'perm_rank', 'shap_rank']].mean(axis=1)
        
        # Sort by average rank
        comparison = comparison.sort_values('avg_rank')
        
        # Add consistency score (lower std = more consistent)
        comparison['rank_std'] = comparison[['tree_rank', 'perm_rank', 'shap_rank']].std(axis=1)
        
        self.logger.info("  ✓ Importance comparison complete")
        
        return comparison
    
    def comprehensive_feature_analysis(
        self,
        ridge_model: Ridge,
        rf_model: RandomForestRegressor,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_std: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run comprehensive feature analysis with all methods.
        
        Parameters
        ----------
        ridge_model : Ridge
            Trained Ridge model
        rf_model : RandomForestRegressor
            Trained Random Forest model
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        feature_std : Optional[np.ndarray]
            Feature standard deviations
            
        Returns
        -------
        Dict with all analysis results
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("COMPREHENSIVE FEATURE ANALYSIS")
        self.logger.info("=" * 70)
        
        results = {}
        
        # 1. Linear coefficients
        self.logger.info("\n1. Linear Model Coefficients")
        results['linear_coef'] = self.compute_linear_coefficients(ridge_model, feature_std)
        
        # 2. Tree importance
        self.logger.info("\n2. Tree-Based Importance")
        results['tree_importance'] = self.compute_tree_importance(rf_model)
        
        # 3. Permutation importance (on test set)
        self.logger.info("\n3. Permutation Importance (Ridge)")
        results['perm_importance_ridge'] = self.compute_permutation_importance(
            ridge_model, X_test, y_test, n_repeats=10
        )
        
        self.logger.info("\n4. Permutation Importance (Random Forest)")
        results['perm_importance_rf'] = self.compute_permutation_importance(
            rf_model, X_test, y_test, n_repeats=10
        )
        
        # 4. SHAP values
        self.logger.info("\n5. SHAP Values (Random Forest)")
        results['shap_rf'] = self.compute_shap_values(
            rf_model, X_test, model_type='tree', max_samples=1000
        )
        
        # 5. Compare methods
        self.logger.info("\n6. Comparing Importance Methods")
        results['importance_comparison'] = self.compare_importance_methods(
            results['tree_importance'],
            results['perm_importance_rf'],
            results['shap_rf']['feature_importance']
        )
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FEATURE ANALYSIS COMPLETE")
        self.logger.info("=" * 70)
        
        return results
    
    def get_top_features(
        self,
        importance_df: pd.DataFrame,
        n_features: int = 10
    ) -> List[str]:
        """
        Get top N most important features.
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            Feature importance DataFrame
        n_features : int
            Number of top features to return
            
        Returns
        -------
        List of feature names
        """
        return importance_df.head(n_features)['feature'].tolist()
