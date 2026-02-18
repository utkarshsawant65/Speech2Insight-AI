# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

## [0.1.0] - 2024-01-01

### Added

- Streamlit UI with 5-step toggleable pipeline (Upload & Transcribe, Preprocess, Sentiment,
  Topic Modeling, Summarization)
- OpenAI Whisper transcription (tiny/base/small/medium/large); ffmpeg bundled via
  `imageio-ffmpeg` — no manual install required
- NLTK preprocessing: lowercase, punct/digit removal, stopword removal with negative words kept
  for accurate sentiment analysis
- TextBlob chunk-based sentiment (polarity, subjectivity, label)
- Aspect-based sentiment analysis
- Optional emotion detection via `j-hartmann/emotion-english-distilroberta-base`
- LSA topic modeling (TF-IDF + TruncatedSVD) with heatmap and per-topic word clouds
- T5 summarization (`google-t5/t5-base`) with chunking for long text
- Optional BLEU and ROUGE-1/2/L evaluation vs. reference summary
- Docker support (Dockerfile + docker-compose.yml) with system ffmpeg
- GitHub Actions CI: Ruff lint + format check, mypy type check, pytest with coverage,
  Docker build, push to GHCR, pip-audit security scan
- Environment variable support for all pipeline constants (`.env.example`)
- Makefile for developer shortcuts (`install`, `run`, `test`, `lint`, `format`, `docker-up`)
- Pre-commit hooks (Ruff lint + format, trailing whitespace, end-of-file fixer)
- Dependabot configuration for weekly pip and GitHub Actions dependency updates
- CONTRIBUTING.md, SECURITY.md, GitHub issue template, PR template
