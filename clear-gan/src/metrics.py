
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def g_mean_score_from_cm(cm: np.ndarray) -> float:
    """Compute geometric mean of per-class sensitivities from a confusion matrix."""
    cm = cm.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.diag(cm) / np.maximum(cm.sum(axis=1), 1e-12)
    sens = np.nan_to_num(sens, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.prod(sens)))


def g_mean_score(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    return g_mean_score_from_cm(cm)


def class_balance_accuracy(y_true, y_pred) -> float:
    """Class Balance Accuracy (CBA) as implemented in your Colab notebook."""
    cm = confusion_matrix(y_true, y_pred).astype(float)
    num_classes = cm.shape[0]
    # For each true class i, take the max predicted count and normalise by row sum
    row_sums = np.maximum(cm.sum(axis=1), 1e-12)
    per_class = np.max(cm, axis=1) / row_sums
    return float(np.mean(per_class))


def mean_sensitivity(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred).astype(float)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    sens = tp / np.maximum(tp + fn, 1e-12)
    return float(np.mean(sens))


def confusion_entropy(y_true, y_pred) -> float:
    """Confusion Entropy (CEN), normalised to [0, 1], matching your notebook."""
    cm = confusion_matrix(y_true, y_pred).astype(float)
    c = cm.shape[0]
    total = np.maximum(cm.sum(), 1e-12)

    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)

    # Class probability weighting
    pj = cm.sum(axis=1) / (2.0 * total)

    pij = np.maximum(cm / row_sums, 1e-12)

    cen_j = -np.sum(
        pij * np.log2(pij) + (1.0 - pij) * np.log2(np.maximum(1.0 - pij, 1e-12)),
        axis=1,
    )

    cen = float(np.sum(pj * cen_j))
    denom = np.log2(max(c, 2))
    cen_norm = cen / denom
    return float(np.clip(cen_norm, 0.0, 1.0))
