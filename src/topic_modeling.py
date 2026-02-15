"""Task 4: Topic modeling with LSA (TF-IDF + TruncatedSVD), word clouds."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import N_TOPICS, TOPIC_CHUNK_SIZE


def chunk_text(text: str, chunk_size: int = TOPIC_CHUNK_SIZE) -> list[str]:
    """Split text into ~chunk_size word chunks (documents for LSA)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return [c for c in chunks if c.strip()]


def run_lsa(
    documents: list[str],
    n_topics: int = N_TOPICS,
    max_features: int = 2000,
    min_df: int = 1,
    max_df: float = 0.95,
) -> tuple[Any, Any, np.ndarray, list[list[str]], list[list[float]]]:
    """
    LSA: TfidfVectorizer + TruncatedSVD.
    Returns: vectorizer, svd, doc_topic matrix, top words per topic, top weights per topic.
    """
    if not documents:
        return None, None, np.array([]), [], []
    n_topics = min(n_topics, len(documents), max_features)
    if n_topics < 1:
        n_topics = 1
    vectorizer = TfidfVectorizer(
        max_features=max_features, min_df=min_df, max_df=max_df, stop_words="english"
    )
    X = vectorizer.fit_transform(documents)
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    doc_topic = svd.fit_transform(X)
    feature_names = vectorizer.get_feature_names_out()
    top_words_per_topic = []
    top_weights_per_topic = []
    for i in range(n_topics):
        comp = svd.components_[i]
        top_idx = np.argsort(comp)[::-1][:15]
        top_words_per_topic.append([feature_names[j] for j in top_idx])
        top_weights_per_topic.append([float(comp[j]) for j in top_idx])
    return vectorizer, svd, doc_topic, top_words_per_topic, top_weights_per_topic


def topic_heatmap(doc_topic: np.ndarray, n_docs_show: int = 20) -> BytesIO:
    """Heatmap of topic distribution across documents."""
    fig, ax = plt.subplots(figsize=(10, max(4, min(12, doc_topic.shape[0] * 0.3))))
    n_show = min(n_docs_show, doc_topic.shape[0])
    data = doc_topic[:n_show]
    sns.heatmap(
        data.T,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=range(1, n_show + 1),
        yticklabels=[f"Topic {i + 1}" for i in range(data.shape[1])],
    )
    ax.set_xlabel("Document chunk")
    ax.set_ylabel("Topic")
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def wordcloud_for_topic(words: list[str], weights: list[float] | None = None) -> BytesIO:
    """Word cloud for one topic (word list; optional weights)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "wordcloud not available", ha="center", va="center")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf
    if weights and len(weights) == len(words):
        freq = dict(zip(words, weights))
    else:
        freq = {w: 1 for w in words}
    wc = WordCloud(width=400, height=200, background_color="white").generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
