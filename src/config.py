"""Pipeline configuration for speech2insight-AI.

All values can be overridden via environment variables (see .env.example).
"""

import os

# Whisper: tiny | base | small | medium | large
WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")

# Preprocessing — these negative words are always kept during stopword removal
NEGATIVE_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nobody",
    "nothing",
    "nowhere",
    "nor",
    "cannot",
    "can't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    "mightn't",
    "mustn't",
}

# Sentiment (Step 3)
SENTIMENT_CHUNK_SIZE: int = int(os.environ.get("SENTIMENT_CHUNK_SIZE", "200"))
NEUTRAL_THRESHOLD: float = float(os.environ.get("NEUTRAL_THRESHOLD", "0.05"))

# Topic modeling (Step 4)
TOPIC_CHUNK_SIZE: int = int(os.environ.get("TOPIC_CHUNK_SIZE", "300"))
N_TOPICS: int = int(os.environ.get("N_TOPICS", "5"))

# Summarization (Step 5)
SUMMARY_MODEL: str = os.environ.get("SUMMARY_MODEL", "google-t5/t5-base")
SUMMARY_MAX_LENGTH: int = int(os.environ.get("SUMMARY_MAX_LENGTH", "150"))
SUMMARY_MIN_LENGTH: int = int(os.environ.get("SUMMARY_MIN_LENGTH", "50"))
SUMMARY_CHUNK_SIZE: int = int(os.environ.get("SUMMARY_CHUNK_SIZE", "512"))

# Emotion model (Step 3)
EMOTION_MODEL: str = os.environ.get(
    "EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"
)
