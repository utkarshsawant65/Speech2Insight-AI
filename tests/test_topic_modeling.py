"""Tests for topic modeling module."""

import numpy as np

from src.topic_modeling import chunk_text, run_lsa, topic_heatmap, wordcloud_for_topic


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_splits(sample_preprocessed_text: str) -> None:
    chunks = chunk_text(sample_preprocessed_text, chunk_size=5)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


def test_run_lsa_empty_documents() -> None:
    vec, svd, doc_topic, top_words, top_weights = run_lsa([])
    assert vec is None
    assert svd is None
    assert doc_topic.size == 0
    assert not top_words
    assert not top_weights


def test_run_lsa_returns_expected_shapes(sample_docs: list[str]) -> None:
    n_topics = 2
    vec, svd, doc_topic, top_words, top_weights = run_lsa(
        sample_docs, n_topics=n_topics
    )
    assert vec is not None
    assert svd is not None
    assert doc_topic.shape[0] == len(sample_docs)
    assert doc_topic.shape[1] == n_topics
    assert len(top_words) == n_topics
    assert len(top_weights) == n_topics
    assert all(len(w) <= 15 for w in top_words)


def test_topic_heatmap_returns_bytes() -> None:
    doc_topic = np.array([[0.5, 0.5], [0.8, 0.2], [0.2, 0.8]])
    buf = topic_heatmap(doc_topic)
    assert buf is not None
    data = buf.read()
    assert len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_wordcloud_for_topic_returns_bytes() -> None:
    words = ["hello", "world", "test"]
    weights = [1.0, 0.5, 0.3]
    buf = wordcloud_for_topic(words, weights)
    assert buf is not None
    data = buf.read()
    assert len(data) > 0
