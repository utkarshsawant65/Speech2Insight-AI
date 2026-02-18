# speech2insight-AI

[![CI](https://github.com/utkarshsawant65/NLP-Project---audio-to-txt-converter/actions/workflows/ci.yml/badge.svg)](https://github.com/utkarshsawant65/NLP-Project---audio-to-txt-converter/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](LICENSE)

Turn speech into insights: upload an audio file and run a full NLP pipeline —
**Transcribe → Preprocess → Sentiment → Topic Modeling → Summarization** — through a Streamlit web UI.

---

## Demo

The app runs in the browser. After `streamlit run app.py`, open `http://localhost:8501`.

> A screenshot or screen recording of the Streamlit UI can be added here (e.g. `docs/demo.gif`).

---

## Quickstart

### Local

```bash
# 1. Clone
git clone https://github.com/utkarshsawant65/NLP-Project---audio-to-txt-converter.git
cd NLP-Project---audio-to-txt-converter

# 2. Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) copy and edit env vars
cp .env.example .env

# 5. Run
streamlit run app.py
```

ffmpeg is bundled via `imageio-ffmpeg` — no separate install required.

### Docker

```bash
# Build and run (includes ffmpeg and all deps)
docker compose up --build
```

Open `http://localhost:8501`.

Or with plain Docker:

```bash
docker build -t speech2insight-ai .
docker run --env-file .env.example -p 8501:8501 speech2insight-ai
```

---

## How to Run Tests

```bash
pip install -r requirements-test.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
```

Or via Makefile (after `make install`):

```bash
make test
```

Tests avoid loading Whisper or HuggingFace models (mocked) so CI stays fast.

---

## Configuration

Copy `.env.example` to `.env` and adjust values. All settings have sensible defaults and
can also be set as OS environment variables.

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `SENTIMENT_CHUNK_SIZE` | `200` | Words per chunk for TextBlob sentiment |
| `NEUTRAL_THRESHOLD` | `0.05` | Polarity threshold for neutral classification |
| `TOPIC_CHUNK_SIZE` | `300` | Words per chunk for LSA topic modeling |
| `N_TOPICS` | `5` | Default number of LSA topics |
| `SUMMARY_MODEL` | `google-t5/t5-base` | HuggingFace model for summarization |
| `SUMMARY_MAX_LENGTH` | `150` | Max tokens per summary chunk |
| `SUMMARY_MIN_LENGTH` | `50` | Min tokens per summary chunk |
| `SUMMARY_CHUNK_SIZE` | `512` | Words per chunk fed to T5 |
| `EMOTION_MODEL` | `j-hartmann/emotion-english-distilroberta-base` | HuggingFace model for emotion detection |

---

## Project Structure

```
speech2insight-AI/
├── app.py                        # Streamlit UI (5-step pipeline)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                # Ruff, pytest, mypy, coverage config
├── requirements.txt              # Production deps
├── requirements-test.txt         # Minimal deps for fast CI tests
├── Makefile                      # Developer shortcuts
├── .env.example                  # Template for environment variables
├── .pre-commit-config.yaml       # Pre-commit hooks (ruff, whitespace)
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CHANGELOG.md
├── LICENSE
├── .github/
│   ├── workflows/
│   │   └── ci.yml               # Lint, type-check, test, Docker build/push
│   ├── dependabot.yml           # Automated dependency updates
│   ├── ISSUE_TEMPLATE/
│   │   └── bug_report.md
│   └── pull_request_template.md
├── tests/
│   ├── conftest.py
│   ├── test_preprocess.py
│   ├── test_sentiment.py
│   ├── test_summarization.py
│   ├── test_topic_modeling.py
│   └── test_transcribe.py
└── src/
    ├── config.py                 # Env-var-backed pipeline constants
    ├── logger.py
    ├── transcribe.py             # Whisper transcription
    ├── preprocess.py             # NLTK preprocessing
    ├── sentiment.py              # TextBlob + transformers sentiment
    ├── topic_modeling.py         # LSA (TF-IDF + TruncatedSVD)
    └── summarization.py          # T5 summarization + BLEU/ROUGE
```

---

## Architecture

```
Audio file (mp3/wav/m4a/ogg/flac)
        |
        v
[1] Transcribe  -- OpenAI Whisper (tiny/base/small/medium/large)
        |              ffmpeg (system or bundled via imageio-ffmpeg)
        v
[2] Preprocess  -- NLTK tokenization, lowercase, punct/digit removal,
        |              stopword removal (negative words kept for sentiment)
        v
[3] Sentiment   -- TextBlob chunk-based polarity/subjectivity
        |              Aspect-based sentiment (per-aspect sentence scoring)
        |              Optional: emotion detection (transformers pipeline)
        v
[4] Topics      -- LSA: TF-IDF vectorizer + TruncatedSVD
        |              Heatmap (seaborn) + word cloud (wordcloud) per topic
        v
[5] Summarize   -- T5 (google-t5/t5-base) chunked summarization
                   Optional: BLEU / ROUGE-1/2/L vs. reference summary
```

Each step is independently togglable in the sidebar. Heavy models (Whisper, T5, emotion
pipeline) are lazy-loaded and cached on first use.

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):

| Job | Trigger | What it does |
|---|---|---|
| **Lint & Type Check** | Push/PR to `main` or `master` | Ruff lint + format check; mypy on `src/` |
| **Test** | After lint | pytest with coverage >= 60%; Pylint |
| **Security Audit** | Push/PR (independent) | pip-audit on `requirements.txt` |
| **Build** | After test | Docker image build (no push) |
| **Push** | Push to `main`/`master` only | Builds and pushes to GitHub Container Registry |

---

## Known Limitations

- **Whisper model size**: The `base` model (default) is fast but less accurate than `large`.
  Larger models require more RAM and are much slower on CPU. `fp16=False` is set to prevent
  errors on CPU-only machines.
- **GPU**: No GPU is configured in Docker or CI. Inference runs on CPU. For production use,
  a CUDA-enabled image and GPU instance are recommended.
- **ffmpeg**: Bundled via `imageio-ffmpeg` for convenience. If the bundled binary is
  unavailable (rare edge case), install ffmpeg manually and add it to `PATH`.
- **Short audio**: Very short clips (< 5 seconds) may transcribe poorly with smaller Whisper
  models.
- **Topic modeling**: LSA requires at least 2 distinct chunks (`TOPIC_CHUNK_SIZE` words each).
  Very short transcripts may not produce meaningful topics.
- **T5 summarization**: Works best on English text of 50+ words. Very short inputs produce
  low-quality summaries.

---

## Roadmap / Next Steps

- Add speaker diarization (e.g. pyannote-audio)
- Support real-time microphone input via `streamlit-webrtc`
- Expose a REST API alongside the Streamlit UI (FastAPI)
- Add GPU support in Docker (NVIDIA base image)
- Add a persistent transcript/results store (SQLite)
- Expand language support beyond English (Whisper multilingual models)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, code style, and PR process.

```bash
pip install pre-commit
pre-commit install
```

---

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities.

---

## License

This is free and unencumbered software released into the public domain.
See [LICENSE](LICENSE) (Unlicense).
