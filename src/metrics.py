from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Threshold is tuned on VALIDATION only.
    """
    thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.50
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(
                "threshold metric must be: f1, recall, or balanced_accuracy"
            )

        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score


def binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_malignant": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_malignant": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(
            average_precision_score(y_true, probabilities)
        )
    except ValueError:
        metrics["pr_auc"] = float("nan")

    return metrics
