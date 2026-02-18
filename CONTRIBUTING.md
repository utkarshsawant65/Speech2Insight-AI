# Contributing to speech2insight-AI

Thank you for your interest in contributing. This document explains how to set up the
development environment, run tests, and submit changes.

## Dev Setup

```bash
# 1. Fork and clone
git clone https://github.com/utkarshsawant65/NLP-Project---audio-to-txt-converter.git
cd NLP-Project---audio-to-txt-converter

# 2. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# 3. Install all dependencies
make install
# or manually:
pip install -r requirements.txt
pip install -e ".[dev]"
pip install -r requirements-test.txt

# 4. Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Running Tests

```bash
make test
# or:
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
```

Tests avoid loading Whisper or HuggingFace models (mocked in CI). All tests must pass
before submitting a PR.

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
make lint      # check for lint errors
make format    # check formatting (non-destructive)
```

- Line length: 100 characters (configured in `pyproject.toml`)
- Target Python version: 3.10+
- Import order: enforced by Ruff (isort-compatible)

Pre-commit hooks run Ruff automatically on `git commit`. Fix any issues flagged before pushing.

## Type Checking

```bash
mypy src/ --ignore-missing-imports --no-strict-optional
```

Type annotations are expected on new public functions. The project uses `ignore_missing_imports`
because several heavy dependencies (whisper, textblob, wordcloud) have no type stubs.

## Pull Request Process

1. Create a feature branch from `main`: `git checkout -b feat/your-feature`
2. Make changes; ensure `make lint` and `make test` both pass
3. Push and open a PR against `main`
4. Fill in the PR template
5. CI must pass (lint, type check, tests, Docker build) before merging

## Reporting Issues

Use GitHub Issues and fill in the bug report template.

## License

By contributing, you agree your changes will be released under the [Unlicense](LICENSE).
