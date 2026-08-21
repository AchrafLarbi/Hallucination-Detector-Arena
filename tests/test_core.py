"""Offline tests (no model downloads, no network). Run: python -m pytest -q"""

import pytest

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
