"""Tests for preprocessing module."""

from src.preprocess import (
    clean_raw_whisper_text,
    get_effective_stopwords,
    preprocess_document,
    preprocess_for_nlp,
)


def test_clean_raw_whisper_text_empty() -> None:
    assert clean_raw_whisper_text("") == ""
    assert clean_raw_whisper_text("   ") == ""


def test_clean_raw_whisper_text_skips_probability_lines() -> None:
    text = "Hello world.\n0.5 0.3 0.2\nprobability of speech 0.9\nGoodbye."
    out = clean_raw_whisper_text(text)
    assert "Hello world" in out
    assert "Goodbye" in out
    assert "0.5" not in out
    assert "probability" not in out


def test_preprocess_for_nlp_empty() -> None:
    assert preprocess_for_nlp("") == ""
    assert preprocess_for_nlp("   ") == ""


def test_preprocess_for_nlp_lowercase(sample_raw_text: str) -> None:
    out = preprocess_for_nlp(sample_raw_text, remove_stopwords=False)
    assert out == out.lower()
    assert "hello" in out


def test_preprocess_for_nlp_removes_digits(sample_raw_text: str) -> None:
    out = preprocess_for_nlp(sample_raw_text, remove_stopwords=False)
    assert "123" not in out


def test_preprocess_for_nlp_keeps_negative_words() -> None:
    text = "this is not bad and we do not like it"
    out = preprocess_for_nlp(text)
    assert "not" in out.split()


def test_get_effective_stopwords_excludes_negatives() -> None:
    sw = get_effective_stopwords()
    assert "not" not in sw
    assert "the" in sw or "is" in sw


def test_preprocess_document_roundtrip(sample_raw_text: str) -> None:
    out = preprocess_document(sample_raw_text, clean_whisper=False)
    assert isinstance(out, str)
    assert len(out) > 0
    assert out == out.lower()


def test_preprocess_for_nlp_non_string_returns_empty() -> None:
    """Non-string input is rejected; returns empty string."""
    # type ignore intentional for robustness test
    out = preprocess_for_nlp(123)  # type: ignore[arg-type]
    assert out == ""  # isinstance(text, str) is False
