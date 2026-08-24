"""Evaluation statistics. Positive class = hallucinated (1); detectors output support (high = faithful)."""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score


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


def bootstrap_ci(metric, y, values, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of metric(y, values), resampling examples with replacement."""
    rng = np.random.default_rng(seed)
    y, values = np.asarray(y), np.asarray(values)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], values[idx]))
    if not stats:  # e.g. a single class: the metric is undefined
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar_p(correct_a, correct_b) -> float:
    """Exact McNemar test on paired correctness (two-sided)."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if only_a + only_b == 0:
        return 1.0
    return float(binomtest(only_a, only_a + only_b, 0.5).pvalue)


def kappa(pred_a, pred_b) -> float:
    return float(cohen_kappa_score(pred_a, pred_b))
