# FPL Optimiser: Backend

Pulls Fantasy Premier League data, predicts expected points per player, optimises a user's
squad and transfers, and emails the recommendation. Serves an API for the React frontend.
Designed to stay under ~£2/month on GCP: serverless, scale-to-zero, no always-on resources.

## Architecture

```
Scheduler -> Workflows -> Cloud Run Jobs: ingest -> predict -> optimise -> notify
                                   |
                           Cloud Storage (Parquet)
                                   |
web -> FastAPI (Cloud Run service) -> reads predictions, runs optimiser per user
```

Shared, expensive work (ingest + predict) runs once per gameweek. Per-user work (fetch squad,
optimise) is cheap and runs on demand.

| Component | Status |
|---|---|
| `common/` config, logging, Parquet I/O, FPL client | done |
| `ingest/` FPL API -> Parquet | done |
| `predict/`, `optimise/`, `notify/`, `api/` | not started |
| `.github/` CI and deploy | not started |

## Data layout

`gs://$DATA_BUCKET/gw=<n>/` (or `./data/gw=<n>/` locally):

- `players`, `teams`, `events`, `fixtures`: snapshot taken when ingest ran for gameweek `n`.
  The players snapshot holds FPL's `ep_next`, which cannot be recovered later and is the
  baseline for backtests.
- `live`: per-player stats for gameweek `n` once it has finished.

Don't use `GAMEWEEK=<past>` to re-snapshot an old gameweek: it would store *today's* players
data under that gameweek.

## Local development

Requires [uv](https://docs.astral.sh/uv/). `make` targets wrap these commands (on Windows,
run the `uv run ...` commands directly).

```
uv sync                                   # make setup
uv run ruff check . && uv run ruff format --check . && uv run mypy .   # make lint
uv run pytest                             # make test
DRY_RUN=true uv run python -m ingest.main # fixture data, no network
uv run python -m ingest.main              # live FPL API -> ./data
```

Copy `.env.example` to `.env` for local settings. No secrets are committed.

## Repos

- `fpl-optimiser-infra`: all GCP infrastructure (Terraform, Workflows, Scheduler, IAM).
- `fpl-optimiser-web`: React frontend.

## Backtest results

None yet. Results, including if the model fails to beat `ep_next`, will be reported in `docs/`.

`archive/` holds the original single-file optimiser, kept for reference only.
