"""Task 2: Preprocessing — clean, lowercase, remove punct/digits, stopwords (keep negatives)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Set

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .config import NEGATIVE_WORDS


# Ensure NLTK data (report: punkt, stopwords)
def _ensure_nltk_data():
    for name in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{name}")
        except LookupError:
            nltk.download(name, quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def get_effective_stopwords() -> Set[str]:
    """Stopwords minus negative words (for sentiment)."""
    _ensure_nltk_data()
    sw = set(stopwords.words("english"))
    return sw - NEGATIVE_WORDS


def clean_raw_whisper_text(text: str) -> str:
    """Remove Whisper artifacts (e.g. probability lines, timestamps). Trim after end line."""
    if not text or not text.strip():
        return ""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that look like stats/probabilities
        if re.match(r"^[\d.\s%]+$", line):
            continue
        if line.lower().startswith("probability") or "probability of" in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines)


def preprocess_for_nlp(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_digits: bool = True,
    remove_special: bool = True,
    remove_stopwords: bool = True,
    custom_stopwords: Set[str] | None = None,
) -> str:
    """
    Clean and tokenize: lowercase, remove punctuation/digits/special, stopwords (keeping negatives).
    Returns space-joined tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    _ensure_nltk_data()
    if custom_stopwords is None:
        custom_stopwords = get_effective_stopwords()

    t = text
    if lowercase:
        t = t.lower()
    if remove_punct:
        t = re.sub(r"[^\w\s]", " ", t)
    if remove_digits:
        t = re.sub(r"\d+", " ", t)
    if remove_special:
        t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    tokens = word_tokenize(t)
    if remove_stopwords:
        tokens = [w for w in tokens if w not in custom_stopwords and len(w) > 1]
    return " ".join(tokens)


def preprocess_document(
    raw_text: str,
    clean_whisper: bool = True,
    **preprocess_kw,
) -> str:
    """Full pipeline: optional Whisper cleanup + preprocess_for_nlp."""
    if clean_whisper:
        raw_text = clean_raw_whisper_text(raw_text)
    return preprocess_for_nlp(raw_text, **preprocess_kw)


def save_text(content: str, path: str | Path) -> Path:
    """Save string to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
