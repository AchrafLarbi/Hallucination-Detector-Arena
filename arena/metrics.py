"""Evaluation statistics. Positive class = hallucinated (1); detectors output support (high = faithful)."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def predict(support, threshold: float) -> np.ndarray:
    """Hallucinated (1) when support < threshold."""
    return (np.asarray(support) < threshold).astype(int)


def auroc(y, support) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, 1.0 - np.asarray(support))) if len(set(y)) == 2 else float("nan")


def balanced_accuracy(y, pred) -> float:
    return float(balanced_accuracy_score(y, pred))


def f1_hallucinated(y, pred) -> float:
    return float(f1_score(y, pred, pos_label=1, zero_division=0))


def best_threshold(y, support) -> float:
    """Threshold maximising balanced accuracy (tuned on the dev split only)."""
    s = np.unique(np.asarray(support))
    candidates = np.concatenate([[0.0], (s[:-1] + s[1:]) / 2, [1.0 + 1e-9]]) if len(s) > 1 else np.array([0.5])
    scores = [balanced_accuracy(y, predict(support, t)) for t in candidates]
    return float(candidates[int(np.argmax(scores))])
