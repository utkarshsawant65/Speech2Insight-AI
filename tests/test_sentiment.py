"""Tests for sentiment module."""

from src.sentiment import (
    SentimentResult,
    aspect_based_sentiment,
    chunk_text,
    sentiment_chunked,
)


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_splits_by_size() -> None:
    words = " ".join(["w"] * 250)
    chunks = chunk_text(words, chunk_size=100)
    assert len(chunks) == 3
    assert all(len(c.split()) <= 100 for c in chunks)


def test_sentiment_chunked_empty() -> None:
    res = sentiment_chunked("")
    assert res == SentimentResult(0.0, 0.0, "neutral")


def test_sentiment_chunked_returns_named_tuple() -> None:
    res = sentiment_chunked("I love this. It is great and amazing.")
    assert isinstance(res, SentimentResult)
    assert res.polarity >= -1 and res.polarity <= 1
    assert res.subjectivity >= 0 and res.subjectivity <= 1
    assert res.label in ("positive", "negative", "neutral")


def test_sentiment_chunked_positive_text() -> None:
    res = sentiment_chunked("Great amazing wonderful fantastic!")
    assert res.label == "positive"


def test_aspect_based_sentiment_empty() -> None:
    assert not aspect_based_sentiment("", ["love"])
    assert not aspect_based_sentiment("hello world", [])


def test_aspect_based_sentiment_returns_scores() -> None:
    text = "The food was good. The service was bad. The food again was okay."
    aspects = ["food", "service"]
    out = aspect_based_sentiment(text, aspects)
    assert "food" in out
    assert "service" in out
    assert all(isinstance(v, float) for v in out.values())
