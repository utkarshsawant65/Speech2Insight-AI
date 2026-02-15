"""
NLP Audio-to-Text Pipeline — Streamlit UI
End-to-end: Upload Audio → Transcribe → Preprocess → Sentiment → Topic Modeling → Summarization
Based on DA2 NLP Mini Project Report.
"""

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import (
    N_TOPICS,
    SENTIMENT_CHUNK_SIZE,
    SUMMARY_MAX_LENGTH,
    SUMMARY_MIN_LENGTH,
    TOPIC_CHUNK_SIZE,
)
from src.preprocess import preprocess_document, preprocess_for_nlp
from src.sentiment import (
    aspect_based_sentiment,
    get_emotions_transformers,
    sentiment_chunked,
)
from src.summarization import bleu_score, rouge_scores, summarize_with_t5
from src.topic_modeling import chunk_text as topic_chunk_text
from src.topic_modeling import run_lsa, topic_heatmap, wordcloud_for_topic
from src.transcribe import load_whisper_model, transcribe_uploaded_file

st.set_page_config(page_title="NLP Audio-to-Text Pipeline", layout="wide")

st.title("NLP Audio-to-Text Pipeline")
st.caption(
    "End-to-end: Audio → Transcribe → Preprocess → Sentiment → Topics → Summarization (DA2 NLP)"
)

# Session state for pipeline outputs
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = ""
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None

# Sidebar: pipeline steps
st.sidebar.header("Pipeline")
step_upload = st.sidebar.checkbox("1. Upload & Transcribe", value=True)
step_preprocess = st.sidebar.checkbox("2. Preprocess", value=True)
step_sentiment = st.sidebar.checkbox("3. Sentiment", value=True)
step_topics = st.sidebar.checkbox("4. Topic Modeling", value=True)
step_summary = st.sidebar.checkbox("5. Summarization", value=True)

# ----- 1. Upload & Transcribe -----
if step_upload:
    st.header("1. Upload & Transcribe (Whisper)")
    col1, col2 = st.columns([2, 1])
    with col1:
        audio_file = st.file_uploader(
            "Upload audio (mp3, wav, m4a, ...)", type=["mp3", "wav", "m4a", "ogg", "flac"]
        )
    with col2:
        whisper_model_name = st.selectbox(
            "Whisper model", ["tiny", "base", "small", "medium", "large"], index=1
        )

    if audio_file:
        if st.button("Transcribe", key="transcribe_btn"):
            with st.spinner(
                "Loading Whisper model and transcribing… (may take a while for long audio)"
            ):
                try:
                    model = load_whisper_model(whisper_model_name)
                    st.session_state.whisper_model = model
                    text = transcribe_uploaded_file(
                        audio_file, model=model, model_name=whisper_model_name
                    )
                    st.session_state.transcript = text
                    st.success("Transcription done.")
                except Exception as e:
                    st.error(str(e))
        if st.session_state.transcript:
            st.subheader("Transcript")
            st.text_area(
                "Raw transcript", st.session_state.transcript, height=200, key="transcript_ta"
            )

    # Allow pasting text instead of audio (for demo / pre-recorded transcript)
    st.subheader("Or paste transcript")
    pasted = st.text_area("Paste text to run rest of pipeline", height=100, key="pasted_transcript")
    if pasted and st.button("Use pasted text as transcript"):
        st.session_state.transcript = pasted
        st.rerun()

# ----- 2. Preprocess -----
if step_preprocess and st.session_state.transcript:
    st.header("2. Preprocess")
    preprocessed = preprocess_document(st.session_state.transcript)
    st.session_state.preprocessed = preprocessed
    st.text_area(
        "Preprocessed text (cleaned, tokenized, stopwords removed, negatives kept)",
        preprocessed,
        height=150,
        key="preproc_ta",
    )

# ----- 3. Sentiment -----
if step_sentiment and st.session_state.transcript:
    st.header("3. Sentiment Analysis")
    text_for_sentiment = st.session_state.preprocessed or preprocess_for_nlp(
        st.session_state.transcript
    )
    if text_for_sentiment:
        res = sentiment_chunked(text_for_sentiment, chunk_size=SENTIMENT_CHUNK_SIZE)
        c1, c2, c3 = st.columns(3)
        c1.metric("Polarity", f"{res.polarity:.3f}")
        c2.metric("Subjectivity", f"{res.subjectivity:.3f}")
        c3.metric("Label", res.label)
        st.caption("Chunk-based TextBlob; neutral threshold 0.05")

        # Aspect-based (optional: user can add aspects)
        aspects_input = st.text_input(
            "Aspect-based sentiment — comma-separated aspects (e.g. god, love, faith)",
            key="aspects",
        )
        if aspects_input:
            aspects = [a.strip() for a in aspects_input.split(",") if a.strip()]
            if aspects:
                absa = aspect_based_sentiment(st.session_state.transcript, aspects)
                st.write("Aspect polarities:", absa)

        # Emotions (transformers) — optional, can be slow
        if st.checkbox("Run emotion detection (transformers)", value=False, key="run_emotions"):
            with st.spinner("Loading emotion model…"):
                emotions = get_emotions_transformers(text_for_sentiment)
            if emotions:
                st.bar_chart(emotions)

# ----- 4. Topic Modeling -----
if step_topics and st.session_state.transcript:
    st.header("4. Topic Modeling (LSA)")
    text_for_topic = st.session_state.preprocessed or preprocess_for_nlp(
        st.session_state.transcript
    )
    if text_for_topic:
        n_topics = st.slider("Number of topics", 2, 10, N_TOPICS, key="n_topics")
        docs = topic_chunk_text(text_for_topic, chunk_size=TOPIC_CHUNK_SIZE)
        if len(docs) >= 1:
            try:
                vec, svd, doc_topic, top_words, top_weights = run_lsa(docs, n_topics=n_topics)
                st.subheader("Topic heatmap")
                buf = topic_heatmap(doc_topic)
                st.image(buf)
                st.subheader("Top words per topic")
                tabs = st.tabs([f"Topic {i + 1}" for i in range(n_topics)])
                for i, (words, weights) in enumerate(zip(top_words, top_weights)):
                    with tabs[i]:
                        st.write("Top words:", ", ".join(words[:10]))
                        wc_buf = wordcloud_for_topic(words, weights)
                        st.image(wc_buf)
            except Exception as e:
                st.warning(f"LSA error: {e} (need enough distinct documents)")
        else:
            st.info("Need more text (chunk size 300 words) for topic modeling.")

# ----- 5. Summarization -----
if step_summary and st.session_state.transcript:
    st.header("5. Summarization (T5)")
    full_text = st.session_state.transcript
    if len(full_text.split()) > 50:
        if st.button("Generate summary", key="summarize_btn"):
            with st.spinner("Summarizing with T5…"):
                summary = summarize_with_t5(
                    full_text,
                    max_length=SUMMARY_MAX_LENGTH,
                    min_length=SUMMARY_MIN_LENGTH,
                )
                st.session_state.summary = summary
        if st.session_state.get("summary"):
            st.subheader("Summary")
            st.write(st.session_state.summary)
            # Optional: BLEU/ROUGE if user provides reference
            ref = st.text_area("Reference summary (optional, for BLEU/ROUGE)", key="ref_summary")
            if ref and st.session_state.get("summary"):
                bleu = bleu_score(ref, st.session_state.summary)
                rouge = rouge_scores(ref, st.session_state.summary)
                st.metric("BLEU", f"{bleu:.4f}")
                st.json(rouge)
    else:
        st.info("Transcript is short; add more text for summarization.")

# Footer
st.sidebar.divider()
st.sidebar.caption("DA2 NLP Mini Project — Whisper, NLTK, TextBlob, LSA, T5")
