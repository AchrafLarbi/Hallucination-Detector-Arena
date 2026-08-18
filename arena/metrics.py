"""Evaluation statistics. Positive class = hallucinated (1); detectors output support (high = faithful)."""

import numpy as np
from sklearn.metrics import roc_auc_score


def predict(support, threshold: float) -> np.ndarray:
    """Hallucinated (1) when support < threshold."""
    return (np.asarray(support) < threshold).astype(int)


def auroc(y, support) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, 1.0 - np.asarray(support))) if len(set(y)) == 2 else float("nan")
