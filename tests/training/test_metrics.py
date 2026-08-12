import numpy as np

from veritarach.training.metrics import compute_metrics


def test_compute_metrics_on_perfect_predictions():
    logits = np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])  # -> [0, 1, 0, 1]
    labels = np.array([0, 1, 0, 1])

    result = compute_metrics((logits, labels))

    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_compute_metrics_on_all_wrong_predictions():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])  # -> [0, 1]
    labels = np.array([1, 0])  # both wrong

    result = compute_metrics((logits, labels))

    assert result["accuracy"] == 0.0
    assert result["f1"] == 0.0


def test_compute_metrics_on_known_mixed_case():
    # predictions: [1, 1, 0, 0] vs labels: [1, 0, 0, 0]
    # TP=1 (idx0), FP=1 (idx1), TN=2 (idx2,3), FN=0
    logits = np.array([[0.0, 10.0], [0.0, 10.0], [10.0, 0.0], [10.0, 0.0]])
    labels = np.array([1, 0, 0, 0])

    result = compute_metrics((logits, labels))

    assert result["accuracy"] == 0.75
    assert result["precision"] == 0.5  # 1 TP / (1 TP + 1 FP)
    assert result["recall"] == 1.0  # 1 TP / (1 TP + 0 FN)
