.PHONY: setup lint test pipeline train backtest api

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy .

test:
	uv run pytest

# ingest -> train -> predict locally against ./data (add stages as they land).
# In production `train` runs on its own schedule, not every gameweek.
pipeline:
	uv run python -m ingest.main
	uv run python -m train.main
	uv run python -m predict.main

train:
	uv run python -m train.main

backtest:
	uv run python -m ml.backtest --no-ablation

api:
	@echo "api not implemented yet"
