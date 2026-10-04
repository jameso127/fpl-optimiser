.PHONY: setup lint test pipeline api

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy .

test:
	uv run pytest

# ingest -> predict -> optimise locally against ./data (add stages as they land)
pipeline:
	uv run python -m ingest.main

api:
	@echo "api not implemented yet"
