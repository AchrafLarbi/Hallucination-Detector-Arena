"""Offline tests (no model downloads, no network). Run: python -m pytest -q"""

import numpy as np
import pandas as pd
import pytest

from arena import metrics as M
from arena.detectors import parse_judge
from arena.study import analyze
from arena.text import chunk_by_tokens, make_claim, make_sentence_claims, split_sentences


# ---------------- text ----------------
def test_split_sentences_keeps_decimals_and_abbreviations():
    text = "Revenue was 2.5 billion, e.g. from cloud. It grew 10%! Was it audited? Yes.\nNew line here"
    assert split_sentences(text) == [
        "Revenue was 2.5 billion, e.g. from cloud.", "It grew 10%!", "Was it audited?", "Yes.", "New line here"]


def test_claims_include_the_question_for_short_answers():
    assert make_claim("What is the ratio?", "0.25") == "What is the ratio? 0.25"
    assert make_claim("", "Paris.") == "Paris."
    assert make_sentence_claims("Q?", "A one. B two.") == ["Q? A one.", "Q? B two."]


def test_chunk_by_tokens_packs_sentences_under_budget():
    text = "One two three. Four five six. Seven eight nine."
    chunks = chunk_by_tokens(text, lambda s: len(s.split()), max_tokens=6)
    assert chunks == ["One two three. Four five six.", "Seven eight nine."]
    assert chunk_by_tokens("", lambda s: 0, 5) == [""]


# ---------------- judge parsing ----------------
@pytest.mark.parametrize("raw, expected", [
    ('{"support_probability": 85, "reason": "ok"}', 0.85),
    ('Sure:\n```json\n{"support_probability": 0, "reason": "contradiction"}\n```', 0.0),
    ('{"support_probability": "ninety", "reason": "seen in the wild"}', 0.90),
    ('{"support_probability": "ninety-five"}', 0.95),
    ('{"support_probability": "one hundred"}', 1.0),
    ('{"support_probability": "70%"}', 0.70),
])
def test_parse_judge(raw, expected):
    assert parse_judge(raw)[0] == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["no json here", '{"support_probability": 150}', '{"reason": "x"}'])
def test_parse_judge_rejects_unusable_answers(raw):
    with pytest.raises((ValueError, KeyError)):
        parse_judge(raw)


# ---------------- metrics ----------------
def test_auroc_and_predictions_use_hallucinated_as_positive():
    y = np.array([1, 1, 0, 0])
    support = np.array([0.1, 0.2, 0.8, 0.9])  # low support = hallucinated
    assert M.auroc(y, support) == 1.0
    assert list(M.predict(support, 0.5)) == [1, 1, 0, 0]


def test_best_threshold_maximises_balanced_accuracy():
    y = np.array([1, 1, 1, 0, 0, 0])
    support = np.array([0.1, 0.2, 0.3, 0.35, 0.8, 0.9])
    t = M.best_threshold(y, support)
    assert 0.3 < t <= 0.35 and M.balanced_accuracy(y, M.predict(support, t)) == 1.0


def test_mcnemar_and_bootstrap():
    assert M.mcnemar_p([1, 1, 1], [1, 1, 1]) == 1.0
    assert M.mcnemar_p([1] * 12, [0] * 12) < 0.001
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 50)
    s = np.clip(0.5 - 0.3 * y + rng.normal(0, 0.2, 100), 0, 1)
    lo, hi = M.bootstrap_ci(M.auroc, y, s, n=300)
    assert lo <= M.auroc(y, s) <= hi


# ---------------- study analysis ----------------
def _fake_study():
    rows, scores = [], []
    rng = np.random.default_rng(1)
    for split in ("dev", "test"):
        for src in ("A", "B"):
            for i in range(20):
                y = i % 2
                rid = f"{split}-{src}-{i}"
                rows.append({"id": rid, "source": src, "split": split, "hallucinated": y})
                # "good" detector: separates well but its natural cut-off is 0.3, not 0.5
                scores.append({"id": rid, "detector": "good", "support": 0.2 if y else 0.4 + rng.uniform(0, 0.1),
                               "seconds": 0.1, "error": ""})
                scores.append({"id": rid, "detector": "coin", "support": rng.uniform(), "seconds": 0.1, "error": ""})
    return pd.DataFrame(rows), pd.DataFrame(scores)


def test_analyze_calibrates_on_dev_and_reports_on_test():
    sample, scores = _fake_study()
    r = analyze(sample, scores, {"good": "Good", "coin": "Coin"})
    good = r["detectors"]["good"]
    assert good["test_auroc"] == 1.0
    assert good["test_bal_acc_default"] == 0.5  # everything below 0.5 -> all flagged
    assert good["test_bal_acc_calibrated"] == 1.0  # threshold learnt on dev transfers to test
    assert 0.2 < good["calibrated_threshold"] < 0.4
    assert set(good["test_per_source_auroc"]) == {"A", "B"}
    assert r["pairwise"][0]["n"] == 40


def test_partially_scored_detectors_are_not_compared():
    sample, scores = _fake_study()
    partial = scores[scores["detector"] == "good"].head(30).assign(detector="judge")
    r = analyze(sample, pd.concat([scores, partial]), {"good": "Good", "coin": "Coin", "judge": "Judge"})
    assert "judge" not in r["detectors"]
    assert r["in_progress"]["judge"] == {"name": "Judge", "n_scored": 30, "n_total": 80}
    assert all("judge" not in (p["a"], p["b"]) for p in r["pairwise"])
