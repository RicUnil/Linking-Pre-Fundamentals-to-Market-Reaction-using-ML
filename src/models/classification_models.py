"""
Classification model training and evaluation utilities.

This module provides functions to train binary classification models
and compute standard classification metrics including ROC-AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """
    Container for classification metrics on a given dataset split.
    
    Attributes
    ----------
    accuracy : float
        Overall accuracy (correct predictions / total).
    balanced_accuracy : float
        Average of recall for each class (handles class imbalance).
    precision : float
        Precision for the positive class (1).
    recall : float
        Recall for the positive class (1).
    f1 : float
        F1-score for the positive class (1).
    roc_auc : float
        Area under the ROC curve (main metric for classification).
    """

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> ClassificationMetrics:
    """
    Compute a standard set of classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted class labels (0/1).
    y_proba : np.ndarray or None
        Predicted probability for the positive class (label=1).
        If None, ROC-AUC is set to NaN.

    Returns
    -------
    ClassificationMetrics
        Container with accuracy, balanced accuracy, precision,
        recall, F1-score, and ROC-AUC.
        
    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> y_proba = np.array([0.1, 0.8, 0.4, 0.2])
    >>> metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    >>> metrics.accuracy
    0.75
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            # Handle edge case where only one class is present
            auc = float("nan")
    else:
        auc = float("nan")

    return ClassificationMetrics(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc,
    )


def train_classification_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Train several classification models on the training set and evaluate them
    on train, validation, and test splits.

    Models trained:
    - dummy_most_frequent: Baseline (always predicts most frequent class)
    - logistic_regression: L2-regularized logistic regression
    - random_forest_classifier: Random forest with 300 trees
    - gradient_boosting_classifier: Gradient boosting with 200 estimators

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Feature matrices for train, validation, and test splits.
    y_train, y_val, y_test : np.ndarray
        Binary classification labels (0/1) for each split.

    Returns
    -------
    models : dict
        Dictionary mapping model_name -> fitted model instance.
    metrics : dict
        Nested dictionary:
        {
            model_name: {
                "train": {metric_name: value, ...},
                "val": {metric_name: value, ...},
                "test": {metric_name: value, ...},
            },
            ...
        }
        
    Examples
    --------
    >>> X_train = np.random.randn(100, 5)
    >>> y_train = np.random.randint(0, 2, 100)
    >>> X_val = np.random.randn(20, 5)
    >>> y_val = np.random.randint(0, 2, 20)
    >>> X_test = np.random.randn(20, 5)
    >>> y_test = np.random.randint(0, 2, 20)
    >>> models, metrics = train_classification_models(
    ...     X_train, y_train, X_val, y_val, X_test, y_test
    ... )
    >>> "logistic_regression" in models
    True
    >>> "roc_auc" in metrics["logistic_regression"]["val"]
    True
    """
    models: Dict[str, Any] = {}

    # 1) Dummy baseline (most frequent class)
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    models["dummy_most_frequent"] = dummy

    # 2) Logistic Regression
    logreg = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    logreg.fit(X_train, y_train)
    models["logistic_regression"] = logreg

    # 3) Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    rf_clf.fit(X_train, y_train)
    models["random_forest_classifier"] = rf_clf

    # 4) Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gb_clf.fit(X_train, y_train)
    models["gradient_boosting_classifier"] = gb_clf

    # === EVALUATION ===
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    def _predict_all(model: Any, name: str) -> Dict[str, Dict[str, float]]:
        """Evaluate a model on all splits."""
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Predicted probabilities for positive class (if available)
        if hasattr(model, "predict_proba"):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            # Map decision_function to probabilities via sigmoid-like assumption.
            # For ROC-AUC, decision_function outputs are fine.
            df_train = model.decision_function(X_train)
            df_val = model.decision_function(X_val)
            df_test = model.decision_function(X_test)
            y_train_proba = df_train
            y_val_proba = df_val
            y_test_proba = df_test
        else:
            y_train_proba = y_val_proba = y_test_proba = None

        train_metrics = compute_classification_metrics(
            y_true=y_train,
            y_pred=y_train_pred,
            y_proba=y_train_proba,
        ).to_dict()
        val_metrics = compute_classification_metrics(
            y_true=y_val,
            y_pred=y_val_pred,
            y_proba=y_val_proba,
        ).to_dict()
        test_metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
        ).to_dict()

        return {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

    for name, model in models.items():
        metrics[name] = _predict_all(model, name)

    return models, metrics
