"""Tests for summarization module (BLEU, ROUGE, chunking; no heavy T5)."""

from src.summarization import bleu_score, chunk_for_summary, rouge_scores


def test_chunk_for_summary_empty() -> None:
    assert chunk_for_summary("") == []
    assert chunk_for_summary("   ") == []


def test_chunk_for_summary_splits() -> None:
    words = " ".join(["w"] * 600)
    chunks = chunk_for_summary(words, chunk_size=256)
    assert len(chunks) >= 2
    assert all(len(c.split()) <= 256 for c in chunks)


def test_bleu_score_identical() -> None:
    ref = "the cat sat on the mat"
    cand = "the cat sat on the mat"
    score = bleu_score(ref, cand)
    assert score >= 0.99


def test_bleu_score_different() -> None:
    ref = "the cat sat on the mat"
    cand = "the dog ran in the park"
    score = bleu_score(ref, cand)
    assert 0 <= score <= 1


def test_bleu_score_empty_candidate() -> None:
    ref = "hello world"
    cand = ""
    score = bleu_score(ref, cand)
    assert 0 <= score <= 1


def test_rouge_scores_returns_dict() -> None:
    ref = "the cat sat on the mat"
    cand = "the cat sat on the mat"
    scores = rouge_scores(ref, cand)
    assert isinstance(scores, dict)
    if scores:
        assert "rouge1" in scores or "rouge2" in scores or "rougeL" in scores
        assert all(0 <= v <= 1 for v in scores.values())
