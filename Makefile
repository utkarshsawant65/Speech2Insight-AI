# speech2insight-AI — Developer shortcuts
# Usage: make <target>
# Requires: Python 3.10+, pip, docker compose

.PHONY: install run test lint format docker-up docker-down clean

## install: Install all production + dev + test dependencies
install:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pip install -r requirements-test.txt

## run: Start the Streamlit app locally
run:
	streamlit run app.py

## test: Run pytest with coverage
test:
	python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('punkt_tab', quiet=True)"
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

## lint: Run ruff linter on all source files
lint:
	ruff check app.py src/ tests/

## format: Check formatting with ruff (non-destructive)
format:
	ruff format --check app.py src/ tests/

## docker-up: Build and start containers with docker compose
docker-up:
	docker compose up --build

## docker-down: Stop and remove containers
docker-down:
	docker compose down

## clean: Remove Python cache and build artifacts
clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov .pytest_cache .mypy_cache dist build *.egg-info
