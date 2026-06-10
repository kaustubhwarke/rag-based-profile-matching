"""Low-level text normalisation and tokenisation helpers."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")

# Minimal English stopword set — sufficient for keyword/BM25 hygiene without
# pulling in a heavyweight NLP dependency.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have in into is it its of on or that the
    to was were will with we you your our their they this these those then than
    """.split()
)


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and excessive blank lines."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Tokenise into lowercase alphanumeric terms (keeps c++, c#, node.js)."""

    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def truncate(text: str, max_chars: int = 280) -> str:
    """Truncate text to ``max_chars`` on a word boundary, adding an ellipsis."""

    text = text.strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0]
    return f"{clipped.rstrip()}…"
