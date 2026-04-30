"""
Generic behavioral indicators for glitch classification (paper §8.3).

The original Glitcher classifier checks for hardcoded model-specific strings
("edreader", "referentialaction", …) and so generalises poorly across models
and tokenizers. This module provides model-agnostic detectors:

    - length_anomaly: response length far below or above prompt-conditioned
      norms.
    - ngram_repetition: dominant repeating substring or n-gram looping.
    - topic_drift: low lexical overlap between prompt content words and
      response (a cheap, dependency-free proxy for semantic drift).
    - refusal_pattern: regex-matched refusal phrases.

Each detector is a closure-friendly callable: pass parameters at construction
time, then call the returned function with `(response, prompt)`. The
classifier code uses thin lambdas so individual tests can pin custom
thresholds.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Iterable, List, Optional, Sequence


# A small stopword list is sufficient for content-word extraction; we do not
# need full NLP machinery.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "for", "to",
    "in", "on", "at", "by", "with", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "i",
    "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "do", "does", "did", "have", "has", "had", "what", "which", "who", "how",
    "why", "when", "where", "from", "into", "about", "than", "so", "such",
    "not", "no", "yes", "please", "can", "will", "would", "could", "should",
    "your", "my", "our", "their", "his", "hers", "ours", "theirs",
})

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 1]


# --- Length anomaly -------------------------------------------------------------


def length_anomaly(
    *,
    min_chars: int = 0,
    max_chars: Optional[int] = None,
    min_ratio: float = 0.0,
    max_ratio: Optional[float] = None,
) -> Callable[[str, str], bool]:
    """
    Build a detector that flags responses whose length is anomalous.

    `min_chars`/`max_chars` are absolute bounds. `min_ratio`/`max_ratio` are
    relative to the prompt length (response_chars / prompt_chars). Either or
    both pairs can be supplied; the detector triggers if any bound is
    violated.
    """
    def _check(response: str, prompt: str = "") -> bool:
        n = len(response or "")
        if n < min_chars:
            return True
        if max_chars is not None and n > max_chars:
            return True
        if prompt:
            ratio = n / max(len(prompt), 1)
            if ratio < min_ratio:
                return True
            if max_ratio is not None and ratio > max_ratio:
                return True
        return False
    return _check


# --- N-gram repetition ----------------------------------------------------------


def _word_ngrams(words: Sequence[str], n: int) -> List[tuple]:
    if n < 1 or len(words) < n:
        return []
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def ngram_repetition(
    *,
    n: int = 3,
    min_total_ngrams: int = 8,
    max_top_share: float = 0.4,
) -> Callable[[str, str], bool]:
    """
    Detector: flag responses where one n-gram dominates >= `max_top_share`
    of all n-grams (e.g. a phrase looping). `min_total_ngrams` guards
    against false positives on very short outputs.
    """
    def _check(response: str, prompt: str = "") -> bool:
        words = _tokenize(response or "")
        grams = _word_ngrams(words, n)
        if len(grams) < min_total_ngrams:
            return False
        top = Counter(grams).most_common(1)[0][1]
        return (top / len(grams)) >= max_top_share
    return _check


def char_repetition(
    *,
    min_chars: int = 30,
    max_top_share: float = 0.6,
) -> Callable[[str, str], bool]:
    """
    Detector: flag responses dominated by a single character (e.g. ``aaaa…``
    or whitespace flooding). Helpful for IDOS-style hangs.
    """
    def _check(response: str, prompt: str = "") -> bool:
        text = response or ""
        if len(text) < min_chars:
            return False
        most_common_char, count = Counter(text).most_common(1)[0]
        return (count / len(text)) >= max_top_share
    return _check


# --- Topic drift ---------------------------------------------------------------


def topic_drift(
    *,
    min_overlap: float = 0.05,
    min_prompt_content_words: int = 3,
) -> Callable[[str, str], bool]:
    """
    Detector: flag responses whose lexical overlap with the prompt's content
    words is below `min_overlap` (Jaccard-style). Cheap proxy for semantic
    drift; replace with an embedding-based comparator when latency budget
    allows.
    """
    def _check(response: str, prompt: str = "") -> bool:
        prompt_words = set(_content_tokens(prompt))
        if len(prompt_words) < min_prompt_content_words:
            # Not enough signal in the prompt to call this anomalous.
            return False
        response_words = set(_content_tokens(response))
        if not response_words:
            return True
        overlap = len(prompt_words & response_words) / len(prompt_words | response_words)
        return overlap < min_overlap
    return _check


# --- Refusal pattern -----------------------------------------------------------


_REFUSAL_RE = re.compile(
    r"\b("
    r"i\s+(?:can(?:not|'t)|won'?t|will\s+not|am\s+(?:not\s+able|unable))"
    r"|sorry,?\s+(?:i|but)"
    r"|i'?m\s+(?:not\s+able|unable)"
    r"|as\s+an?\s+(?:ai|language\s+model|assistant)"
    r"|i\s+(?:must|cannot)\s+(?:decline|refuse)"
    r"|that\s+would\s+(?:not\s+be|be\s+inappropriate)"
    r"|against\s+(?:my|the)\s+(?:policy|guidelines)"
    r")\b",
    re.IGNORECASE,
)


def refusal_pattern(*, additional_patterns: Iterable[str] = ()) -> Callable[[str, str], bool]:
    """
    Detector: flag responses that match a curated set of refusal phrases.
    `additional_patterns` are appended as regex alternatives.
    """
    extra = "|".join(additional_patterns)
    if extra:
        compiled = re.compile(_REFUSAL_RE.pattern + "|" + extra, re.IGNORECASE)
    else:
        compiled = _REFUSAL_RE

    def _check(response: str, prompt: str = "") -> bool:
        return bool(compiled.search(response or ""))
    return _check


# --- Convenience -------------------------------------------------------------


def any_indicator(*detectors: Callable[[str, str], bool]) -> Callable[[str, str], bool]:
    """Combine detectors with logical OR."""
    def _check(response: str, prompt: str = "") -> bool:
        return any(d(response, prompt) for d in detectors)
    return _check


def all_indicators(*detectors: Callable[[str, str], bool]) -> Callable[[str, str], bool]:
    """Combine detectors with logical AND."""
    def _check(response: str, prompt: str = "") -> bool:
        return all(d(response, prompt) for d in detectors)
    return _check
