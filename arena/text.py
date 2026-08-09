"""Text helpers: sentence splitting, claim construction, token-budget chunking."""

import re

# Split after . ! ? followed by whitespace and an uppercase letter / digit / quote, or at newlines.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")


def split_sentences(text: str) -> list:
    return [s.strip() for s in _SENT_BOUNDARY.split(text or "") if s and s.strip()]
