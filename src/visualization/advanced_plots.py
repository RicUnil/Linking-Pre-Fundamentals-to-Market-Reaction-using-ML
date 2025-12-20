"""
Advanced Visualization Module

Creates publication-quality multi-panel figures for:
- Statistical test results (bootstrap CIs, hypothesis tests)
- Feature importance comparisons
- Residual diagnostics
- Sector and regime analysis

Author: Academic Project Enhancement
Date: December 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class AdvancedVisualizer:
    """
    Create publication-quality multi-panel figures.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save figures
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def plot_bootstrap_ci(
        self,
        bootstrap_results: Dict,
        model_name: str,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot bootstrap distribution and confidence interval for R².
        
        Parameters
        ----------
        bootstrap_results : Dict
            Results from bootstrap_r2_ci
        model_name : str
            Name of the model
        ax : Optional[plt.Axes]
            Matplotlib axes (creates new if None)
            
        Returns
        -------
        plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        
        # Extract data
        r2_original = bootstrap_results['r2']
        ci_lower = bootstrap_results['ci_lower']
        ci_upper = bootstrap_results['ci_upper']
        bootstrap_r2s = bootstrap_results['bootstrap_r2s']
        
        # Plot histogram
        ax.hist(bootstrap_r2s, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add vertical lines
        ax.axvline(r2_original, color='red', linestyle='--', linewidth=2, label=f'Original R² = {r2_original:.4f}')
        ax.axvline(ci_lower, color='green', linestyle=':', linewidth=2, label=f'95% CI Lower = {ci_lower:.4f}')
        ax.axvline(ci_upper, color='green', linestyle=':', linewidth=2, label=f'95% CI Upper = {ci_upper:.4f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Labels
        ax.set_xlabel('R² Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name}: Bootstrap Distribution of R²')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_feature_importance_comparison(
        self,
        comparison_df: pd.DataFrame,
        top_n: int = 15,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot feature importance comparison across methods.
        
        Parameters
        ----------
        comparison_df : pd.DataFrame
            Feature importance comparison
        top_n : int
            Number of top features to show
        ax : Optional[plt.Axes]
            Matplotlib axes
            
        Returns
        -------
        plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top N features
        top_features = comparison_df.head(top_n)
        
        # Prepare data for plotting
        features = top_features['feature'].values
        tree_rank = top_features['tree_rank'].values
        perm_rank = top_features['perm_rank'].values
        shap_rank = top_features['shap_rank'].values
        
        # Set up positions
        x = np.arange(len(features))
        width = 0.25
        
        # Create bars
        ax.barh(x - width, tree_rank, width, label='Tree-based', color='#1f77b4', alpha=0.8)
        ax.barh(x, perm_rank, width, label='Permutation', color='#ff7f0e', alpha=0.8)
        ax.barh(x + width, shap_rank, width, label='SHAP', color='#2ca02c', alpha=0.8)
        
        # Customize
        ax.set_yticks(x)
        ax.set_yticklabels(features)
        ax.set_xlabel('Rank (lower = more important)')
        ax.set_title(f'Top {top_n} Features: Importance Ranking Comparison')
        ax.legend(loc='best', frameon=True)
        ax.invert_xaxis()  # Lower rank on left
        ax.grid(True, alpha=0.3, axis='x')
        
        return ax
    
    def plot_residual_diagnostics(
        self,
        residuals: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> plt.Figure:
        """
        Create 4-panel residual diagnostic plot.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
        y_pred : np.ndarray
            Predicted values
        model_name : str
            Name of the model
            
        Returns
        -------
        plt.Figure
        """
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Residuals vs Fitted
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_pred, residuals, alpha=0.5, s=10, color='steelblue')
        ax1.axhline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Q-Q Plot
        ax2 = fig.add_subplot(gs[0, 1])
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Normal Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Scale-Location (sqrt of standardized residuals)
        ax3 = fig.add_subplot(gs[1, 0])
        standardized_residuals = residuals / np.std(residuals)
        ax3.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5, s=10, color='steelblue')
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('√|Standardized Residuals|')
        ax3.set_title('Scale-Location')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Histogram of Residuals
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
        
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
        
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution of Residuals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(f'{model_name}: Residual Diagnostics', fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_sector_comparison(
        self,
        sector_comparison: pd.DataFrame,
        metric: str = 'r2'
    ) -> plt.Figure:
        """
        Create sector comparison plot.
        
        Parameters
        ----------
        sector_comparison : pd.DataFrame
            Sector comparison results
        metric : str
            Metric to plot ('r2', 'mae', 'rmse')
            
        Returns
        -------
        plt.Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = sector_comparison['model'].unique()
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            model_data = sector_comparison[sector_comparison['model'] == model]
            
            # Sort by metric
            model_data = model_data.sort_values(metric, ascending=False)
            
            # Create bar plot
            sectors = model_data['sector'].values
            values = model_data[metric].values
            
            colors = ['green' if v > 0 else 'red' for v in values]
            ax.barh(sectors, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Add zero line
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
            
            # Customize
            ax.set_xlabel(metric.upper())
            ax.set_title(f'{model.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle(f'Sector-Specific Model Performance ({metric.upper()})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_regime_comparison(
        self,
        regime_comparison: pd.DataFrame,
        metric: str = 'r2'
    ) -> plt.Figure:
        """
        Create regime comparison plot.
        
        Parameters
        ----------
        regime_comparison : pd.DataFrame
            Regime comparison results
        metric : str
            Metric to plot
            
        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Pivot data
        pivot_data = regime_comparison.pivot(index='model', columns='regime', values=metric)
        
        # Create grouped bar plot
        x = np.arange(len(pivot_data.index))
        width = 0.35
        
        ax.bar(x - width/2, pivot_data['bull'], width, label='Bull Market', 
               color='green', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, pivot_data['bear'], width, label='Bear Market', 
               color='red', alpha=0.7, edgecolor='black')
        
        # Add zero line
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        
        # Customize
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Model Performance: Bull vs Bear Markets ({metric.upper()})')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in pivot_data.index])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    def create_statistical_summary_figure(
        self,
        ridge_results: Dict,
        rf_results: Dict
    ) -> plt.Figure:
        """
        Create comprehensive 2x2 statistical summary figure.
        
        Parameters
        ----------
        ridge_results : Dict
            Statistical test results for Ridge
        rf_results : Dict
            Statistical test results for Random Forest
            
        Returns
        -------
        plt.Figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Ridge Bootstrap CI
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_bootstrap_ci(ridge_results['r2_bootstrap'], 'Ridge', ax=ax1)
        
        # Panel 2: RF Bootstrap CI
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_bootstrap_ci(rf_results['r2_bootstrap'], 'Random Forest', ax=ax2)
        
        # Panel 3: Model Comparison (R² with CIs)
        ax3 = fig.add_subplot(gs[1, 0])
        models = ['Ridge', 'Random Forest']
        r2_values = [ridge_results['metrics']['r2'], rf_results['metrics']['r2']]
        ci_lowers = [ridge_results['r2_bootstrap']['ci_lower'], 
                     rf_results['r2_bootstrap']['ci_lower']]
        ci_uppers = [ridge_results['r2_bootstrap']['ci_upper'], 
                     rf_results['r2_bootstrap']['ci_upper']]
        
        x = np.arange(len(models))
        ax3.bar(x, r2_values, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
        ax3.errorbar(x, r2_values, 
                     yerr=[np.array(r2_values) - np.array(ci_lowers),
                           np.array(ci_uppers) - np.array(r2_values)],
                     fmt='none', color='black', capsize=5, linewidth=2)
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.set_ylabel('R²')
        ax3.set_title('Model Performance with 95% Confidence Intervals')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Panel 4: P-values Summary
        ax4 = fig.add_subplot(gs[1, 1])
        test_names = ['Permutation\nTest', 'T-test vs\nBaseline']
        ridge_pvals = [ridge_results['permutation_test']['p_value'],
                       ridge_results['t_test_vs_baseline']['p_value']]
        rf_pvals = [rf_results['permutation_test']['p_value'],
                    rf_results['t_test_vs_baseline']['p_value']]
        
        x = np.arange(len(test_names))
        width = 0.35
        
        ax4.bar(x - width/2, ridge_pvals, width, label='Ridge', 
                color='steelblue', alpha=0.7, edgecolor='black')
        ax4.bar(x + width/2, rf_pvals, width, label='Random Forest', 
                color='coral', alpha=0.7, edgecolor='black')
        ax4.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(test_names)
        ax4.set_ylabel('P-value')
        ax4.set_title('Hypothesis Test Results')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([0, 1])
        
        fig.suptitle('Statistical Analysis Summary', fontsize=16, fontweight='bold')
        
        return fig
    
    def create_feature_importance_figure(
        self,
        linear_coef: pd.DataFrame,
        tree_importance: pd.DataFrame,
        perm_importance: pd.DataFrame,
        shap_importance: pd.DataFrame,
        top_n: int = 10
    ) -> plt.Figure:
        """
        Create comprehensive 2x2 feature importance figure.
        
        Parameters
        ----------
        linear_coef : pd.DataFrame
            Linear coefficients
        tree_importance : pd.DataFrame
            Tree-based importance
        perm_importance : pd.DataFrame
            Permutation importance
        shap_importance : pd.DataFrame
            SHAP importance
        top_n : int
            Number of top features
            
        Returns
        -------
        plt.Figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Linear Coefficients
        ax1 = fig.add_subplot(gs[0, 0])
        top_linear = linear_coef.head(top_n)
        colors = ['green' if c > 0 else 'red' for c in top_linear['coefficient']]
        ax1.barh(top_linear['feature'], top_linear['coefficient'], 
                 color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Coefficient')
        ax1.set_title('Linear Model Coefficients')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Panel 2: Tree Importance
        ax2 = fig.add_subplot(gs[0, 1])
        top_tree = tree_importance.head(top_n)
        ax2.barh(top_tree['feature'], top_tree['importance'], 
                 color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Importance')
        ax2.set_title('Tree-Based Importance')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Panel 3: Permutation Importance
        ax3 = fig.add_subplot(gs[1, 0])
        top_perm = perm_importance.head(top_n)
        ax3.barh(top_perm['feature'], top_perm['importance_mean'], 
                 color='coral', alpha=0.7, edgecolor='black')
        ax3.errorbar(top_perm['importance_mean'], range(len(top_perm)),
                     xerr=top_perm['importance_std'], fmt='none', 
                     color='black', capsize=3)
        ax3.set_xlabel('Importance (with std)')
        ax3.set_title('Permutation Importance')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Panel 4: SHAP Importance
        ax4 = fig.add_subplot(gs[1, 1])
        top_shap = shap_importance.head(top_n)
        ax4.barh(top_shap['feature'], top_shap['shap_importance'], 
                 color='purple', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Mean |SHAP Value|')
        ax4.set_title('SHAP Importance')
        ax4.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle(f'Feature Importance: Top {top_n} Features (4 Methods)', 
                     fontsize=16, fontweight='bold')
        
        return fig
    
    def save_all_figures(
        self,
        ridge_results: Dict,
        rf_results: Dict,
        feature_results: Dict,
        residuals: np.ndarray,
        y_pred: np.ndarray,
        sector_comparison: Optional[pd.DataFrame] = None,
        regime_comparison: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Generate and save all figures.
        
        Parameters
        ----------
        ridge_results : Dict
            Ridge statistical results
        rf_results : Dict
            Random Forest statistical results
        feature_results : Dict
            Feature importance results
        residuals : np.ndarray
            Model residuals
        y_pred : np.ndarray
            Predictions
        sector_comparison : Optional[pd.DataFrame]
            Sector comparison data
        regime_comparison : Optional[pd.DataFrame]
            Regime comparison data
        """
        self.logger.info("Generating publication-quality figures...")
        
        # Figure 1: Statistical Summary
        self.logger.info("  Creating statistical summary figure...")
        fig1 = self.create_statistical_summary_figure(ridge_results, rf_results)
        fig1.savefig(self.output_dir / 'figure_01_statistical_summary.png', 
                     bbox_inches='tight', dpi=300)
        plt.close(fig1)
        
        # Figure 2: Feature Importance
        self.logger.info("  Creating feature importance figure...")
        fig2 = self.create_feature_importance_figure(
            feature_results['linear_coef'],
            feature_results['tree_importance'],
            feature_results['perm_importance_rf'],
            feature_results['shap_rf']['feature_importance'],
            top_n=10
        )
        fig2.savefig(self.output_dir / 'figure_02_feature_importance.png', 
                     bbox_inches='tight', dpi=300)
        plt.close(fig2)
        
        # Figure 3: Residual Diagnostics
        self.logger.info("  Creating residual diagnostics figure...")
        fig3 = self.plot_residual_diagnostics(residuals, y_pred, "Ridge Regression")
        fig3.savefig(self.output_dir / 'figure_03_residual_diagnostics.png', 
                     bbox_inches='tight', dpi=300)
        plt.close(fig3)
        
        # Figure 4: Sector Comparison (if available)
        if sector_comparison is not None:
            self.logger.info("  Creating sector comparison figure...")
            fig4 = self.plot_sector_comparison(sector_comparison, metric='r2')
            fig4.savefig(self.output_dir / 'figure_04_sector_comparison.png', 
                         bbox_inches='tight', dpi=300)
            plt.close(fig4)
        
        # Figure 5: Regime Comparison (if available)
        if regime_comparison is not None:
            self.logger.info("  Creating regime comparison figure...")
            fig5 = self.plot_regime_comparison(regime_comparison, metric='r2')
            fig5.savefig(self.output_dir / 'figure_05_regime_comparison.png', 
                         bbox_inches='tight', dpi=300)
            plt.close(fig5)
        
        self.logger.info(f"  ✓ All figures saved to {self.output_dir}")
