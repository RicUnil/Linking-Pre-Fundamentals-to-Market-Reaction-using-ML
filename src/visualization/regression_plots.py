"""
Regression visualization functions for generating PNG figures.

This module provides plotting utilities for:
- Actual vs predicted scatter plots
- Residual histograms and scatter plots
- Bar charts of metrics by model
- Time-series plots of rolling metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    """
    Ensure that a directory exists.

    Parameters
    ----------
    path : Path
        Directory path to create if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def plot_actual_vs_predicted_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Create a scatter plot of actual vs predicted values with y=x reference line.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values.
    y_pred : np.ndarray
        Model predictions.
    out_path : Path
        Path to the output PNG file.
    title : str
        Plot title.
    """
    _ensure_dir(out_path.parent)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Add y=x reference line
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel("Actual 30d excess return")
    ax.set_ylabel("Predicted 30d excess return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_residuals_histogram(
    residuals: np.ndarray,
    out_path: Path,
    title: str,
    bins: int = 50,
) -> None:
    """
    Plot a histogram of residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Residual values (y_true - y_pred).
    out_path : Path
        Path to the output PNG file.
    title : str
        Plot title.
    bins : int, optional
        Number of histogram bins, by default 50.
    """
    _ensure_dir(out_path.parent)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=bins, edgecolor='black', alpha=0.7)
    
    # Add vertical line at zero
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    
    # Add mean line
    mean_res = residuals.mean()
    ax.axvline(mean_res, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_res:.4f}')
    
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_residuals_vs_predictions(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Plot residuals as a function of predicted values.

    Parameters
    ----------
    y_pred : np.ndarray
        Model predictions.
    residuals : np.ndarray
        Residuals (y_true - y_pred).
    out_path : Path
        Path to the output PNG file.
    title : str
        Plot title.
    """
    _ensure_dir(out_path.parent)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    
    # Add horizontal line at zero
    ax.axhline(0.0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    
    ax.set_xlabel("Predicted 30d excess return")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_bar_metrics_by_model(
    df_metrics: pd.DataFrame,
    metric_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """
    Plot a bar chart of a given metric by model.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Dataframe with at least ['model_name', metric_col].
    metric_col : str
        Column name containing the metric to plot.
    out_path : Path
        Path to the output PNG file.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    """
    _ensure_dir(out_path.parent)

    df_sorted = df_metrics.sort_values(metric_col).copy()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(df_sorted)), df_sorted[metric_col], color='steelblue', edgecolor='black')
    
    # Highlight best model (lowest MAE or highest R²)
    if 'mae' in metric_col.lower() or 'rmse' in metric_col.lower():
        bars[0].set_color('green')
        bars[0].set_alpha(0.7)
    elif 'r2' in metric_col.lower():
        bars[-1].set_color('green')
        bars[-1].set_alpha(0.7)
    
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted["model_name"], rotation=45, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_rolling_metric_over_time(
    df_rolling: pd.DataFrame,
    metric_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    models_to_plot: Sequence[str] | None = None,
) -> None:
    """
    Plot a rolling metric (e.g., test_mae or test_r2) over folds for multiple models.

    Parameters
    ----------
    df_rolling : pd.DataFrame
        Dataframe with columns ['fold_id', 'test_start', 'model_name', metric_col].
    metric_col : str
        Name of the metric column to plot.
    out_path : Path
        Path to the output PNG file.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    models_to_plot : sequence of str, optional
        Subset of model names to plot; if None, plot all unique models.
    """
    _ensure_dir(out_path.parent)

    if models_to_plot is None:
        models = sorted(df_rolling["model_name"].unique())
    else:
        models = [m for m in models_to_plot if m in set(df_rolling["model_name"])]

    # Use test_start as x-axis (convert to datetime if needed)
    df = df_rolling.copy()
    df["test_start"] = pd.to_datetime(df["test_start"])

    # Define colors for consistency
    color_map = {
        'baseline_mean': 'black',
        'baseline_capm': 'gray',
        'ridge': 'blue',
        'random_forest': 'green',
        'xgb_best': 'red',
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for model in models:
        sub = df[df["model_name"] == model].sort_values("test_start")
        color = color_map.get(model, None)
        ax.plot(sub["test_start"], sub[metric_col], marker="o", label=model, 
                linewidth=2, markersize=6, color=color)

    ax.set_xlabel("Test period start")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_comprehensive_model_comparison(
    y_val: np.ndarray,
    y_test: np.ndarray,
    predictions_dict: dict,
    out_path: Path,
    title: str = "Comprehensive Model Comparison",
) -> None:
    """
    Create a 4-panel figure comparing model predictions.
    
    Panels:
    1. Validation: Actual vs Predicted (Ridge)
    2. Test: Actual vs Predicted (Ridge)
    3. Validation: Actual vs Predicted (Random Forest)
    4. Test: Actual vs Predicted (Random Forest)

    Parameters
    ----------
    y_val : np.ndarray
        Validation target values.
    y_test : np.ndarray
        Test target values.
    predictions_dict : dict
        Dictionary with keys like 'ridge_val', 'ridge_test', 'rf_val', 'rf_test'.
    out_path : Path
        Path to the output PNG file.
    title : str
        Overall figure title.
    """
    _ensure_dir(out_path.parent)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Panel 1: Ridge Validation
    ax = axes[0, 0]
    y_pred = predictions_dict['ridge_val']
    ax.scatter(y_val, y_pred, alpha=0.3, s=10)
    min_val = min(y_val.min(), y_pred.min())
    max_val = max(y_val.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel("Actual 30d excess return")
    ax.set_ylabel("Predicted 30d excess return")
    ax.set_title("Ridge — Validation Set (n=2,452)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Panel 2: Ridge Test
    ax = axes[0, 1]
    y_pred = predictions_dict['ridge_test']
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel("Actual 30d excess return")
    ax.set_ylabel("Predicted 30d excess return")
    ax.set_title("Ridge — Test Set (n=7,040)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Panel 3: Random Forest Validation
    ax = axes[1, 0]
    y_pred = predictions_dict['rf_val']
    ax.scatter(y_val, y_pred, alpha=0.3, s=10, color='green')
    min_val = min(y_val.min(), y_pred.min())
    max_val = max(y_val.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel("Actual 30d excess return")
    ax.set_ylabel("Predicted 30d excess return")
    ax.set_title("Random Forest — Validation Set (n=2,452)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Panel 4: Random Forest Test
    ax = axes[1, 1]
    y_pred = predictions_dict['rf_test']
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='green')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel("Actual 30d excess return")
    ax.set_ylabel("Predicted 30d excess return")
    ax.set_title("Random Forest — Test Set (n=7,040)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_comprehensive_residual_analysis(
    y_test: np.ndarray,
    predictions_dict: dict,
    out_path: Path,
    title: str = "Comprehensive Residual Analysis",
) -> None:
    """
    Create a 4-panel figure analyzing residuals.
    
    Panels:
    1. Ridge: Residuals histogram
    2. Ridge: Residuals vs Predicted
    3. Random Forest: Residuals histogram
    4. Random Forest: Residuals vs Predicted

    Parameters
    ----------
    y_test : np.ndarray
        Test target values.
    predictions_dict : dict
        Dictionary with keys like 'ridge_test', 'rf_test'.
    out_path : Path
        Path to the output PNG file.
    title : str
        Overall figure title.
    """
    _ensure_dir(out_path.parent)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Ridge residuals
    ridge_residuals = y_test - predictions_dict['ridge_test']
    ridge_pred = predictions_dict['ridge_test']
    
    # Panel 1: Ridge histogram
    ax = axes[0, 0]
    ax.hist(ridge_residuals, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    ax.axvline(ridge_residuals.mean(), color='darkblue', linestyle='--', linewidth=2, 
               label=f'Mean: {ridge_residuals.mean():.4f}')
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Ridge — Residuals Distribution (Test)\nStd: {ridge_residuals.std():.4f}")
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Panel 2: Ridge residuals vs predicted
    ax = axes[0, 1]
    ax.scatter(ridge_pred, ridge_residuals, alpha=0.3, s=10, color='blue')
    ax.axhline(0.0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    ax.set_xlabel("Predicted 30d excess return")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title("Ridge — Residuals vs Predicted (Test)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Random Forest residuals
    rf_residuals = y_test - predictions_dict['rf_test']
    rf_pred = predictions_dict['rf_test']
    
    # Panel 3: RF histogram
    ax = axes[1, 0]
    ax.hist(rf_residuals, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    ax.axvline(rf_residuals.mean(), color='darkgreen', linestyle='--', linewidth=2, 
               label=f'Mean: {rf_residuals.mean():.4f}')
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Random Forest — Residuals Distribution (Test)\nStd: {rf_residuals.std():.4f}")
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Panel 4: RF residuals vs predicted
    ax = axes[1, 1]
    ax.scatter(rf_pred, rf_residuals, alpha=0.3, s=10, color='green')
    ax.axhline(0.0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    ax.set_xlabel("Predicted 30d excess return")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title("Random Forest — Residuals vs Predicted (Test)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_comprehensive_metrics_comparison(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    out_path: Path,
    title: str = "Comprehensive Metrics Comparison",
) -> None:
    """
    Create a 4-panel figure comparing metrics across models.
    
    Panels:
    1. Validation MAE by model
    2. Test MAE by model
    3. Validation R² by model
    4. Test R² by model

    Parameters
    ----------
    df_val : pd.DataFrame
        Validation metrics with columns ['model_name', 'val_mae', 'val_r2'].
    df_test : pd.DataFrame
        Test metrics with columns ['model_name', 'test_mae', 'test_r2'].
    out_path : Path
        Path to the output PNG file.
    title : str
        Overall figure title.
    """
    _ensure_dir(out_path.parent)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Panel 1: Validation MAE
    ax = axes[0, 0]
    df_sorted = df_val.sort_values('val_mae')
    bars = ax.bar(range(len(df_sorted)), df_sorted['val_mae'], color='steelblue', edgecolor='black')
    bars[0].set_color('green')
    bars[0].set_alpha(0.7)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['model_name'], rotation=45, ha='right')
    ax.set_ylabel("Validation MAE")
    ax.set_title(f"Validation MAE by Model (n={len(df_val)})")
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel 2: Test MAE
    ax = axes[0, 1]
    df_sorted = df_test.sort_values('test_mae')
    bars = ax.bar(range(len(df_sorted)), df_sorted['test_mae'], color='coral', edgecolor='black')
    bars[0].set_color('green')
    bars[0].set_alpha(0.7)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['model_name'], rotation=45, ha='right')
    ax.set_ylabel("Test MAE")
    ax.set_title(f"Test MAE by Model (n={len(df_test)})")
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel 3: Validation R²
    ax = axes[1, 0]
    df_sorted = df_val.sort_values('val_r2', ascending=False)
    bars = ax.bar(range(len(df_sorted)), df_sorted['val_r2'], color='steelblue', edgecolor='black')
    if df_sorted['val_r2'].iloc[0] > 0:
        bars[0].set_color('green')
        bars[0].set_alpha(0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['model_name'], rotation=45, ha='right')
    ax.set_ylabel("Validation R²")
    ax.set_title(f"Validation R² by Model (n={len(df_val)})")
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel 4: Test R²
    ax = axes[1, 1]
    df_sorted = df_test.sort_values('test_r2', ascending=False)
    bars = ax.bar(range(len(df_sorted)), df_sorted['test_r2'], color='coral', edgecolor='black')
    if df_sorted['test_r2'].iloc[0] > 0:
        bars[0].set_color('green')
        bars[0].set_alpha(0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['model_name'], rotation=45, ha='right')
    ax.set_ylabel("Test R²")
    ax.set_title(f"Test R² by Model (n={len(df_test)})")
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_comprehensive_rolling_analysis(
    df_rolling: pd.DataFrame,
    out_path: Path,
    title: str = "Comprehensive Rolling-Window Analysis",
) -> None:
    """
    Create a 4-panel figure analyzing rolling-window performance.
    
    Panels:
    1. Rolling Test MAE over time (all models)
    2. Rolling Test R² over time (all models)
    3. Rolling Test MAE variability (box plot by model)
    4. Model ranking stability (heatmap or count)

    Parameters
    ----------
    df_rolling : pd.DataFrame
        Rolling metrics with columns ['fold_id', 'test_start', 'model_name', 'test_mae', 'test_r2'].
    out_path : Path
        Path to the output PNG file.
    title : str
        Overall figure title.
    """
    _ensure_dir(out_path.parent)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Prepare data
    df = df_rolling.copy()
    df["test_start"] = pd.to_datetime(df["test_start"])
    
    color_map = {
        'baseline_mean': 'black',
        'baseline_capm': 'gray',
        'ridge': 'blue',
        'random_forest': 'green',
        'xgb_best': 'red',
    }
    
    models = ['baseline_mean', 'ridge', 'random_forest', 'xgb_best', 'baseline_capm']
    
    # Panel 1: Rolling MAE over time
    ax = axes[0, 0]
    for model in models:
        sub = df[df["model_name"] == model].sort_values("test_start")
        color = color_map.get(model, None)
        ax.plot(sub["test_start"], sub["test_mae"], marker="o", label=model, 
                linewidth=2, markersize=5, color=color)
    ax.set_xlabel("Test Period")
    ax.set_ylabel("Test MAE")
    ax.set_title("Rolling Test MAE Over Time (2015-2025)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 2: Rolling R² over time
    ax = axes[1, 0]
    for model in models:
        sub = df[df["model_name"] == model].sort_values("test_start")
        color = color_map.get(model, None)
        ax.plot(sub["test_start"], sub["test_r2"], marker="o", label=model, 
                linewidth=2, markersize=5, color=color)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Test Period")
    ax.set_ylabel("Test R²")
    ax.set_title("Rolling Test R² Over Time (2015-2025)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 3: MAE variability by model (box plot)
    ax = axes[0, 1]
    data_for_box = [df[df["model_name"] == m]["test_mae"].values for m in models]
    bp = ax.boxplot(data_for_box, labels=models, patch_artist=True)
    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(color_map.get(model, 'lightblue'))
        patch.set_alpha(0.6)
    ax.set_ylabel("Test MAE")
    ax.set_title("Test MAE Variability Across Folds")
    ax.grid(True, axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 4: Average metrics summary
    ax = axes[1, 1]
    avg_metrics = df.groupby('model_name')[['test_mae', 'test_r2']].mean()
    avg_metrics = avg_metrics.loc[models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, avg_metrics['test_mae'], width, label='Avg MAE', 
                   color='steelblue', edgecolor='black', alpha=0.7)
    bars2 = ax2.bar(x + width/2, avg_metrics['test_r2'], width, label='Avg R²', 
                    color='coral', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Test MAE', color='steelblue')
    ax2.set_ylabel('Average Test R²', color='coral')
    ax.set_title('Average Metrics Across All Folds')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_data_overview(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    out_path: Path,
    title: str = "Dataset Overview and Distribution Analysis",
) -> None:
    """
    Create a 4-panel figure showing dataset overview.
    
    Panels:
    1. Sample sizes by split
    2. Target distribution by split
    3. Target distribution over time
    4. Summary statistics table

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with 'excess_return_30d' and date column.
    df_val : pd.DataFrame
        Validation data.
    df_test : pd.DataFrame
        Test data.
    out_path : Path
        Path to the output PNG file.
    title : str
        Overall figure title.
    """
    _ensure_dir(out_path.parent)
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Sample sizes
    ax1 = fig.add_subplot(gs[0, 0])
    splits = ['Train', 'Validation', 'Test']
    sizes = [len(df_train), len(df_val), len(df_test)]
    colors = ['steelblue', 'orange', 'coral']
    bars = ax1.bar(splits, sizes, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_ylabel('Number of Observations')
    ax1.set_title('Sample Sizes by Split')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(size):,}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 2: Target distribution by split
    ax2 = fig.add_subplot(gs[0, 1])
    target_col = 'excess_return_30d'
    
    ax2.hist(df_train[target_col], bins=50, alpha=0.5, label='Train', color='steelblue', edgecolor='black')
    ax2.hist(df_val[target_col], bins=50, alpha=0.5, label='Validation', color='orange', edgecolor='black')
    ax2.hist(df_test[target_col], bins=50, alpha=0.5, label='Test', color='coral', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero return')
    ax2.set_xlabel('30-day Excess Return')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Target Distribution by Split')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Target over time (if date column exists)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Combine all data with split labels
    df_train_copy = df_train.copy()
    df_val_copy = df_val.copy()
    df_test_copy = df_test.copy()
    
    df_train_copy['split'] = 'Train'
    df_val_copy['split'] = 'Validation'
    df_test_copy['split'] = 'Test'
    
    df_all = pd.concat([df_train_copy, df_val_copy, df_test_copy], ignore_index=True)
    
    # Assuming there's a date column (try common names)
    date_cols = ['earnings_date', 'date', 'Date', 'EARNINGS_DATE']
    date_col = None
    for col in date_cols:
        if col in df_all.columns:
            date_col = col
            break
    
    if date_col:
        df_all[date_col] = pd.to_datetime(df_all[date_col])
        df_all = df_all.sort_values(date_col)
        
        # Plot rolling mean
        window = 100
        rolling_mean = df_all.set_index(date_col)[target_col].rolling(window=window, min_periods=1).mean()
        ax3.plot(rolling_mean.index, rolling_mean.values, linewidth=2, color='darkblue')
        ax3.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Date')
        ax3.set_ylabel(f'30-day Excess Return (Rolling Mean, window={window})')
        ax3.set_title('Target Variable Over Time')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax3.text(0.5, 0.5, 'Date column not found', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Target Variable Over Time (N/A)')
    
    # Panel 4: Summary statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate statistics
    stats_data = []
    for name, df in [('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
        stats_data.append([
            name,
            f"{len(df):,}",
            f"{df[target_col].mean():.4f}",
            f"{df[target_col].std():.4f}",
            f"{df[target_col].min():.4f}",
            f"{df[target_col].max():.4f}",
        ])
    
    # Create table
    table = ax4.table(
        cellText=stats_data,
        colLabels=['Split', 'N', 'Mean', 'Std', 'Min', 'Max'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    colors_table = ['#D9E1F2', '#FFF2CC', '#FCE4D6']
    for i, color in enumerate(colors_table, start=1):
        for j in range(6):
            table[(i, j)].set_facecolor(color)
    
    ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_model_performance_summary(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    df_rolling: pd.DataFrame,
    out_path: Path,
    title: str = "Model Performance Summary Across All Evaluations",
) -> None:
    """
    Create a 4-panel figure summarizing model performance.
    
    Panels:
    1. Validation vs Test MAE comparison
    2. Validation vs Test R² comparison
    3. Rolling MAE statistics (mean, min, max)
    4. Model ranking consistency

    Parameters
    ----------
    df_val : pd.DataFrame
        Validation metrics.
    df_test : pd.DataFrame
        Test metrics.
    df_rolling : pd.DataFrame
        Rolling metrics.
    out_path : Path
        Path to the output PNG file.
    title : str
        Overall figure title.
    """
    _ensure_dir(out_path.parent)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Get common models
    common_models = list(set(df_test['model_name']) & set(df_val['model_name']))
    
    # Panel 1: Val vs Test MAE
    ax = axes[0, 0]
    val_mae = df_val[df_val['model_name'].isin(common_models)].set_index('model_name')['val_mae']
    test_mae = df_test[df_test['model_name'].isin(common_models)].set_index('model_name')['test_mae']
    
    x = np.arange(len(common_models))
    width = 0.35
    
    ax.bar(x - width/2, [val_mae[m] for m in common_models], width, label='Validation', 
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.bar(x + width/2, [test_mae[m] for m in common_models], width, label='Test', 
           color='coral', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('MAE')
    ax.set_title('Validation vs Test MAE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(common_models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel 2: Val vs Test R²
    ax = axes[0, 1]
    val_r2 = df_val[df_val['model_name'].isin(common_models)].set_index('model_name')['val_r2']
    test_r2 = df_test[df_test['model_name'].isin(common_models)].set_index('model_name')['test_r2']
    
    ax.bar(x - width/2, [val_r2[m] for m in common_models], width, label='Validation', 
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.bar(x + width/2, [test_r2[m] for m in common_models], width, label='Test', 
           color='coral', edgecolor='black', alpha=0.7)
    
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Model')
    ax.set_ylabel('R²')
    ax.set_title('Validation vs Test R² Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(common_models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel 3: Rolling MAE statistics
    ax = axes[1, 0]
    rolling_stats = df_rolling.groupby('model_name')['test_mae'].agg(['mean', 'min', 'max'])
    rolling_stats = rolling_stats.loc[common_models]
    
    x = np.arange(len(common_models))
    ax.plot(x, rolling_stats['mean'], marker='o', linewidth=2, markersize=8, label='Mean', color='darkblue')
    ax.fill_between(x, rolling_stats['min'], rolling_stats['max'], alpha=0.3, color='lightblue', label='Min-Max Range')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Test MAE')
    ax.set_title('Rolling Test MAE: Mean and Range Across Folds')
    ax.set_xticks(x)
    ax.set_xticklabels(common_models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Model ranking consistency
    ax = axes[1, 1]
    
    # Count how many times each model ranks in top 3
    rank_counts = {model: 0 for model in common_models}
    
    for fold in df_rolling['fold_id'].unique():
        fold_data = df_rolling[df_rolling['fold_id'] == fold].sort_values('test_mae')
        top3 = fold_data.head(3)['model_name'].values
        for model in top3:
            if model in rank_counts:
                rank_counts[model] += 1
    
    models_sorted = sorted(rank_counts.keys(), key=lambda x: rank_counts[x], reverse=True)
    counts = [rank_counts[m] for m in models_sorted]
    
    bars = ax.barh(models_sorted, counts, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Highlight best
    if counts:
        bars[0].set_color('green')
        bars[0].set_alpha(0.7)
    
    ax.set_xlabel('Number of Times in Top 3')
    ax.set_ylabel('Model')
    ax.set_title(f'Model Ranking Consistency (Top 3, {df_rolling["fold_id"].nunique()} folds)')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (model, count) in enumerate(zip(models_sorted, counts)):
        ax.text(count + 0.1, i, str(count), va='center', fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
