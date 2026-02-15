# NLP Audio-to-Text Pipeline

End-to-end NLP pipeline: **Audio → Transcribe → Preprocess → Sentiment → Topic Modeling → Summarization**, with **Streamlit** as the UI. Based on the DA2 NLP Mini Project Report.

## Features

| Step | Description |
|------|-------------|
| **1. Transcribe** | Upload audio (mp3, wav, etc.); transcribe with **OpenAI Whisper** (tiny/base/small/medium/large). |
| **2. Preprocess** | Clean Whisper output; lowercase; remove punctuation/digits; NLTK tokenization; stopwords removed but **negative words kept** for sentiment. |
| **3. Sentiment** | **TextBlob** chunk-based polarity/subjectivity; **aspect-based** sentiment; optional **emotion** detection (transformers). |
| **4. Topic Modeling** | **LSA** (TF-IDF + TruncatedSVD); heatmap + **word clouds** per topic. |
| **5. Summarization** | **T5** (chunked); optional **BLEU/ROUGE** if you provide a reference summary. |

## Requirements

- **Python 3.10+**
- **ffmpeg** (for Whisper) — [install](https://ffmpeg.org/download.html) and ensure `ffmpeg` is on your PATH.

## Setup

```bash
# Clone and enter project
cd NLP-Project---audio-to-txt-converter

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first run will prompt; or run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Run the app

```bash
streamlit run app.py
```

Then open the URL shown (e.g. `http://localhost:8501`).

## Usage

1. **Upload** an audio file and click **Transcribe**, or **paste** a transcript to skip audio.
2. Preprocessing runs on the transcript; you can inspect **preprocessed** text.
3. **Sentiment**: view polarity/subjectivity/label; optionally add **aspects** and run **emotion** detection.
4. **Topic Modeling**: set number of topics; view heatmap and **word clouds** per topic.
5. **Summarization**: generate T5 summary; optionally paste a **reference** summary to see BLEU/ROUGE.

## Docker

Build and run with Docker (includes ffmpeg and all dependencies):

```bash
# Build and run with Docker Compose
docker compose up --build
```

Then open **http://localhost:8501**.

Or with plain Docker:

```bash
docker build -t nlp-audio-txt .
docker run -p 8501:8501 nlp-audio-txt
```

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):

| Job   | Trigger        | What it does |
|-------|----------------|---------------|
| **Lint** | Push/PR to `main` or `master` | Runs [Ruff](https://docs.astral.sh/ruff/) on `app.py` and `src/`. |
| **Build** | After lint | Builds the Docker image (no push). |
| **Push** | Push to `main`/`master` only | Builds and pushes the image to **GitHub Container Registry** (`ghcr.io/<owner>/<repo>`). |

- **Pull requests**: lint + Docker build only.
- **Push to main**: same, then push image to GHCR (no extra secrets needed; `GITHUB_TOKEN` is used).

To pull the image after push:

```bash
docker pull ghcr.io/<your-username>/NLP-Project---audio-to-txt-converter:latest
```

To extend later: add a deploy job (e.g. deploy to Cloud Run, Azure Container Apps, or a VM) or add `pytest` and run tests in CI.

## Project structure

```
NLP-Project---audio-to-txt-converter/
├── app.py                 # Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml          # Ruff config (and future tooling)
├── requirements.txt
├── README.md
├── .github/workflows/
│   └── ci.yml             # CI/CD: lint, Docker build, push to GHCR
├── Report/
│   └── MINI PROJECT REPORT DA2 NLP.docx
└── src/
    ├── config.py          # Chunk sizes, thresholds, model names
    ├── transcribe.py      # Whisper transcription
    ├── preprocess.py      # NLTK preprocessing (stopwords, negatives)
    ├── sentiment.py       # TextBlob, aspect-based, emotions
    ├── topic_modeling.py  # LSA, TF-IDF, word clouds
    └── summarization.py   # T5, BLEU/ROUGE
```

## License

Unlicense (see [LICENSE](LICENSE)).
