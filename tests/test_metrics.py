import numpy as np

from src.metrics import binary_metrics, find_best_threshold


def test_binary_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = binary_metrics(y, p, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["recall_malignant"] == 1.0
    assert metrics["specificity"] == 1.0


def test_threshold_search_returns_valid_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.45, 0.9])

    threshold, score = find_best_threshold(y, p, metric="f1")

    assert 0.05 <= threshold <= 0.95
    assert 0.0 <= score <= 1.0
