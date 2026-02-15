"""Task 5: Summarization with T5 (chunked), BLEU/ROUGE evaluation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .config import SUMMARY_CHUNK_SIZE, SUMMARY_MAX_LENGTH, SUMMARY_MIN_LENGTH, SUMMARY_MODEL


def chunk_for_summary(text: str, chunk_size: int = SUMMARY_CHUNK_SIZE) -> list[str]:
    """Split by ~chunk_size words so model can process."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return [c for c in chunks if c.strip()]


def summarize_with_t5(
    text: str,
    model_name: str = SUMMARY_MODEL,
    max_length: int = SUMMARY_MAX_LENGTH,
    min_length: int = SUMMARY_MIN_LENGTH,
    chunk_size: int = SUMMARY_CHUNK_SIZE,
) -> str:
    """
    Summarize long text by chunking, summarizing each chunk, then joining.
    """
    chunks = chunk_for_summary(text, chunk_size)
    if not chunks:
        return ""
    pipe = _get_summarization_pipeline(model_name)
    if isinstance(pipe, str):
        return pipe  # error message
    summaries = []
    for c in chunks:
        inp = c[:1024]
        if not inp.strip():
            continue
        out = pipe(inp, max_length=max_length, min_length=min_length, do_sample=False)
        if out and isinstance(out, list) and len(out) > 0:
            summaries.append(out[0].get("summary_text", "").strip())
    return " ".join(summaries).strip()


@lru_cache(maxsize=1)
def _get_summarization_pipeline(model_name: str) -> Any:
    """Cached summarization pipeline to avoid reloading on every call."""
    try:
        from transformers import pipeline
        return pipeline("summarization", model=model_name)
    except Exception as e:
        return f"[Model load error: {e}]"


def bleu_score(reference: str, candidate: str) -> float:
    """BLEU between reference and candidate (sentence level)."""
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import sentence_bleu

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        ref_tokens = [word_tokenize(reference)]
        can_tokens = word_tokenize(candidate)
        return float(sentence_bleu(ref_tokens, can_tokens))
    except Exception:
        return 0.0


def rouge_scores(reference: str, candidate: str) -> dict[str, float]:
    """ROUGE-1/2/L F1 (if rouge-score installed)."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {k: v.fmeasure for k, v in scores.items()}
    except ImportError:
        return {}
    except Exception:
        return {}
