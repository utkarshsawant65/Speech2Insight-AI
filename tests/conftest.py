"""Pytest fixtures and shared test data."""

import pytest


@pytest.fixture
def sample_raw_text() -> str:
    """Sample transcript-like text for preprocessing and sentiment."""
    return (
        "Hello world. This is a sample transcript with some punctuation! "
        "It has numbers 123 and special chars. Not bad at all. "
        "We love NLP and sentiment analysis."
    )


@pytest.fixture
def sample_preprocessed_text() -> str:
    """Pre-tokenized style text for topic modeling (no punctuation)."""
    return (
        "hello world sample transcript punctuation numbers special "
        "love nlp sentiment analysis sample text topic modeling"
    )


@pytest.fixture
def sample_docs() -> list[str]:
    """List of short documents for LSA (avoids loading huge vocab)."""
    return [
        "machine learning and deep learning are popular",
        "natural language processing uses machine learning",
        "deep learning models need large data",
    ]
