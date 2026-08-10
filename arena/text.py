"""Text helpers: sentence splitting, claim construction, token-budget chunking."""

import re

# Split after . ! ? followed by whitespace and an uppercase letter / digit / quote, or at newlines.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")


def split_sentences(text: str) -> list:
    return [s.strip() for s in _SENT_BOUNDARY.split(text or "") if s and s.strip()]


def make_claim(question: str, answer: str) -> str:
    """Short answers ("0.25", "Rams") are uncheckable alone: the claim is question + answer."""
    question, answer = (question or "").strip(), (answer or "").strip()
    return f"{question} {answer}".strip() if question else answer


def make_sentence_claims(question: str, answer: str) -> list:
    """One claim per answer sentence (MiniCheck checks sentences, not whole answers)."""
    sentences = split_sentences(answer) or [(answer or "").strip()]
    return [make_claim(question, s) for s in sentences]
