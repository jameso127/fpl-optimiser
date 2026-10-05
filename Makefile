.PHONY: setup lint test pipeline train backtest

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy .

test:
	uv run pytest

# ingest -> train -> predict -> optimise -> notify locally against ./data.
# Needs FPL_TEAM_ID, TELEGRAM_CHAT_ID and TELEGRAM_BOT_TOKEN (see .env.example).
# In production `train` runs on its own schedule, not every gameweek.
pipeline:
	uv run python -m ingest.main
	uv run python -m train.main
	uv run python -m predict.main
	uv run python -m optimise.main
	uv run python -m notify.main

train:
	uv run python -m train.main

backtest:
	uv run python -m ml.backtest --no-ablation
