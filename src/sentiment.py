"""Task 3: Sentiment (TextBlob chunk-based, aspect-based, emotion via transformers)."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, NamedTuple

from textblob import TextBlob

from .config import NEUTRAL_THRESHOLD, SENTIMENT_CHUNK_SIZE


class SentimentResult(NamedTuple):
    polarity: float
    subjectivity: float
    label: str  # positive | negative | neutral


def chunk_text(text: str, chunk_size: int = SENTIMENT_CHUNK_SIZE) -> list[str]:
    """Split text into ~chunk_size word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return [c for c in chunks if c.strip()]


def sentiment_chunked(
    text: str, chunk_size: int = SENTIMENT_CHUNK_SIZE, neutral_threshold: float = NEUTRAL_THRESHOLD
) -> SentimentResult:
    """
    Chunk-based sentiment with TextBlob; average polarity; classify by threshold.
    """
    if not text or not text.strip():
        return SentimentResult(0.0, 0.0, "neutral")
    chunks = chunk_text(text, chunk_size)
    if not chunks:
        return SentimentResult(0.0, 0.0, "neutral")
    polarities = []
    subjectivities = []
    for c in chunks:
        blob = TextBlob(c)
        polarities.append(blob.sentiment.polarity)
        subjectivities.append(blob.sentiment.subjectivity)
    avg_pol = sum(polarities) / len(polarities)
    avg_subj = sum(subjectivities) / len(subjectivities)
    if avg_pol > neutral_threshold:
        label = "positive"
    elif avg_pol < -neutral_threshold:
        label = "negative"
    else:
        label = "neutral"
    return SentimentResult(avg_pol, avg_subj, label)


def aspect_based_sentiment(text: str, aspects: list[str]) -> dict[str, float]:
    """
    For each aspect word, compute average polarity of sentences containing it.
    aspects: list of terms to score (e.g. from report: extracted aspects).
    """
    if not text or not aspects:
        return {}
    # Simple sentence split
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    result = {}
    for aspect in aspects:
        aspect_lower = aspect.lower().strip()
        pols = []
        for sent in sentences:
            if aspect_lower in sent.lower():
                pols.append(TextBlob(sent).sentiment.polarity)
        result[aspect] = sum(pols) / len(pols) if pols else 0.0
    return result


@lru_cache(maxsize=2)
def _get_emotion_pipeline(model_name: str) -> Any | None:
    """Cached emotion pipeline to avoid reloading on every call."""
    try:
        from transformers import pipeline
        return pipeline("text-classification", model=model_name, top_k=None)
    except Exception:
        return None


def get_emotions_transformers(
    text: str,
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    max_length: int = 512,
) -> dict[str, float]:
    """
    Emotion scores using transformers pipeline.
    Returns dict emotion -> score (averaged over chunks if text is long).
    """
    pipe = _get_emotion_pipeline(model_name)
    if pipe is None:
        return {}
    chunks = chunk_text(text, chunk_size=max_length)
    if not chunks:
        return {}
    all_scores = []
    for c in chunks:
        if not c.strip():
            continue
        out = pipe(c[:512], truncation=True, max_length=512)
        if out and isinstance(out, list) and len(out) > 0:
            item = out[0] if not isinstance(out[0], list) else out
            if isinstance(item, list):
                all_scores.append({x["label"]: x["score"] for x in item})
            else:
                all_scores.append({item["label"]: item["score"]})
    if not all_scores:
        return {}
    labels = set()
    for s in all_scores:
        labels.update(s.keys())
    result = {}
    for label in labels:
        vals = [s.get(label, 0.0) for s in all_scores if label in s]
        result[label] = sum(vals) / len(vals) if vals else 0.0
    return result
