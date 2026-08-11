"""Text helpers: sentence splitting, claim construction, token-budget chunking."""

import re

# Split after . ! ? followed by whitespace and an uppercase letter / digit / quote, or at newlines.
# Avoids splitting "e.g. the", "Dr. Smith" (lowercase follows) and decimals like "2.5".
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


def chunk_by_tokens(text: str, count_tokens, max_tokens: int) -> list:
    """Greedy sentence packing into chunks of at most `max_tokens` tokens (MiniCheck's procedure).

    A single sentence longer than the budget becomes its own chunk (the model truncates it).
    """
    chunks, current, current_len = [], [], 0
    for sent in split_sentences(text) or [text or ""]:
        n = count_tokens(sent)
        if current and current_len + n > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [sent], n
        else:
            current.append(sent)
            current_len += n
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()] or [""]


def truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    return text if len(words) <= max_words else " ".join(words[:max_words])
