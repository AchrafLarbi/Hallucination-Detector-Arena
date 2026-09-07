"""Offline tests (no model downloads, no network). Run: python -m pytest -q"""

import pytest

from arena.detectors import parse_judge
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
from arena import metrics as M


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
