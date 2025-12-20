"""
Classification visualization utilities.

This module provides functions to generate classification-related plots:
- ROC curves for multiple models
- Confusion matrices
- Bar plots of AUC scores by model
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix


def _ensure_dir(path: Path) -> None:
    """
    Ensure that a directory exists.

    Parameters
    ----------
    path : Path
        Directory path to create if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def plot_roc_curves_multi_model(
    y_true: np.ndarray,
    proba_by_model: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    """
    Plot ROC curves for multiple models on the same figure.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    proba_by_model : dict
        Mapping from model name to predicted probability (or score)
        for the positive class (label 1).
    out_path : Path
        Path to the output PNG file.
    title : str
        Plot title.
        
    Examples
    --------
    >>> y_true = np.array([0, 1, 0, 1])
    >>> proba_by_model = {
    ...     "model_a": np.array([0.1, 0.8, 0.3, 0.9]),
    ...     "model_b": np.array([0.2, 0.7, 0.4, 0.85]),
    ... }
    >>> plot_roc_curves_multi_model(y_true, proba_by_model, Path("roc.png"), "ROC")
    """
    _ensure_dir(out_path.parent)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Random baseline (diagonal line)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Random (AUC = 0.500)")

    # Plot each model's ROC curve
    for model_name, y_score in proba_by_model.items():
        if y_score is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        model_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {model_auc:.3f})", linewidth=2)

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    normalize: bool = False,
    labels: Sequence[str] | None = None,
) -> None:
    """
    Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    out_path : Path
        Path to the output PNG file.
    title : str
        Plot title.
    normalize : bool, optional
        Whether to normalize the confusion matrix, by default False.
    labels : sequence of str, optional
        Class label names (for display).
        
    Examples
    --------
    >>> y_true = np.array([0, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 1, 0])
    >>> plot_confusion_matrix(y_true, y_pred, Path("cm.png"), "Confusion Matrix")
    """
    _ensure_dir(out_path.parent)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    if labels is None:
        labels = ["0", "1"]

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Annotate each cell with value
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar_auc_by_model(
    df_metrics: pd.DataFrame,
    split: str,
    out_path: Path,
    title: str,
) -> None:
    """
    Plot a bar chart of ROC-AUC by model for a given split.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Metrics dataframe with columns ['model_name', 'split', 'roc_auc'].
    split : str
        Name of the split to filter on ('train', 'val', 'test').
    out_path : Path
        Path to the output PNG file.
    title : str
        Plot title.
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "model_name": ["a", "b"],
    ...     "split": ["val", "val"],
    ...     "roc_auc": [0.5, 0.6],
    ... })
    >>> plot_bar_auc_by_model(df, "val", Path("auc.png"), "AUC by Model")
    """
    _ensure_dir(out_path.parent)

    df_split = df_metrics[df_metrics["split"] == split].copy()
    if df_split.empty:
        return

    df_split = df_split.sort_values("roc_auc", ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create horizontal bar chart for better readability
    bars = ax.barh(df_split["model_name"], df_split["roc_auc"], color="steelblue")
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_split.iterrows()):
        ax.text(
            row["roc_auc"] + 0.01,
            i,
            f"{row['roc_auc']:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    
    # Add reference line at 0.5 (random guessing)
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=2, label="Random (0.500)")
    
    ax.set_xlabel("ROC-AUC", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0.45, max(0.65, df_split["roc_auc"].max() + 0.05))
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
