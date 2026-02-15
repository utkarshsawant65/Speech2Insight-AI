"""Pipeline configuration (aligned with DA2 NLP report)."""

# Whisper: tiny | base | small | medium | large (report used turbo → we use base/small for speed)
WHISPER_MODEL = "base"

# Preprocessing
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

# Sentiment (Task 3)
SENTIMENT_CHUNK_SIZE = 200
NEUTRAL_THRESHOLD = 0.05

# Topic modeling (Task 4)
TOPIC_CHUNK_SIZE = 300
N_TOPICS = 5

# Summarization (Task 5)
SUMMARY_MODEL = "google-t5/t5-base"
SUMMARY_MAX_LENGTH = 150
SUMMARY_MIN_LENGTH = 50
SUMMARY_CHUNK_SIZE = 512

# Emotion model (Task 3)
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
