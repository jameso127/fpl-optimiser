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
| `ingest/` FPL API -> Parquet; `ingest.history` one-off import of past seasons | done |
| `predict/` expected points (hurdle model by default; `ep_next` optional) + backtest | done |
| `optimise/`, `notify/`, `api/` | not started |
| `.github/` CI and deploy | not started |

## Data layout

`gs://$DATA_BUCKET/season=<yyyy-yy>/gw=<n>/` (or `./data/...` locally):

- `players`, `teams`, `events`, `fixtures`: snapshot taken when ingest ran for gameweek `n`.
  The players snapshot holds FPL's `ep_next`, which cannot be recovered later and is the
  baseline for backtests.
- `live`: per-player stats for gameweek `n` once it has finished.
- `predictions`: per player for gameweek `n`: `xpts` (what the optimiser maximises), `ep_next`,
  `xpts_model`, `availability`, `fix_n`.

Don't use `GAMEWEEK=<past>` to re-snapshot an old gameweek: it would store *today's* players
data under that gameweek.

Past seasons (2022-23 onwards) are imported once from the community archive
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) with
`uv run python -m ingest.history`. Its `xP` becomes `ep_next` in those snapshots, stored against the *following* gameweek because the
archive captured it after each gameweek's matches; gameweeks where
the archive recorded all zeros are stored as null. Plan: Parquet in GCS stays the source of
truth, with BigQuery external tables over it later (needs an infra PR).

## Expected points

`XPTS_SOURCE=model` (default) uses our hurdle model: chance of playing x points if playing
(LightGBM), from player form and per-90 rates, team and opponent form, fixtures, rest days and
last season's rates, scaled by FPL's injury news. `XPTS_SOURCE=ep_next` uses FPL's own figure
instead: `form x chance_of_playing / 100`, already doubled in a double gameweek, so it is used
as is. FPL's figure has no fixture awareness, which is the main thing the model adds.

## Local development

Requires [uv](https://docs.astral.sh/uv/). `make` targets wrap these commands (on Windows,
run the `uv run ...` commands directly).

```
uv sync                                   # make setup
uv run ruff check . && uv run ruff format --check . && uv run mypy .   # make lint
uv run pytest                             # make test
DRY_RUN=true uv run python -m ingest.main # fixture data, no network (writes ./data/dry_run)
uv run python -m ingest.main              # live FPL API -> ./data
uv run python -m ingest.history           # one-off: past seasons -> ./data
uv run python -m predict.main             # train + score the latest snapshot
uv run python -m predict.backtest         # regenerate docs/backtest.md
```

Copy `.env.example` to `.env` for local settings. No secrets are committed.

## Repos

- `fpl-optimiser-infra`: all GCP infrastructure (Terraform, Workflows, Scheduler, IAM).
- `fpl-optimiser-web`: React frontend.

## Backtest results

**Preliminary.** The latest full report (`docs/backtest.md`) predates the model rebuild; the
full re-run was interrupted by low memory and has not been repeated yet. What has been
measured for the rebuilt model, on the 9 gameweeks of 2025-26 with a clean pre-deadline
`ep_next` (walk-forward, trained only on earlier data, from a scratch script):

| Same 9 gameweeks | model | `ep_next` |
|---|---|---|
| Top-30 picks, avg points (all / regular players) | 4.52 / 4.46 | 3.93 / 3.80 |
| Spearman (all / regular) | 0.738 / 0.297 | 0.672 / 0.187 |
| MAE (all / regular) | 0.978 / 2.290 | 1.141 / 2.673 |

The model's top-30 beat `ep_next`'s in 7 of 9 gameweeks. Small sample, so treat it as
encouraging, not proven; the full 36-gameweek run and a feature-group ablation are still to
come (`uv run python -m predict.backtest`, or `--no-ablation` for the lighter version).

- `ep_next` is just recent form scaled by availability, with no fixture awareness.
- The archive's raw `xP` was captured *after* each gameweek and contains the outcome. The
  importer stores it against the following gameweek, and the backtest re-runs a leak audit
  every time. An earlier claim that FPL's `xP` beat the model was this leak.

`archive/` holds the original single-file optimiser, kept for reference only.
