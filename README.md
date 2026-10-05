# FPL Optimiser: Backend

Pulls Fantasy Premier League data, predicts expected points per player, optimises a user's
squad and transfers, and emails the recommendation. Serves an API for the React frontend.
Designed to stay under ~£2/month on GCP: serverless, scale-to-zero, no always-on resources.

## Architecture

```
Scheduler -> Workflows -> Cloud Run Jobs: ingest -> predict -> optimise -> notify   (each gameweek)
             Scheduler -> Cloud Run Job:  train                                      (own schedule)
                                   |
                  Cloud Storage (Parquet data, model registry, monitoring)
                                   |
web -> FastAPI (Cloud Run service) -> reads predictions, runs optimiser per user
```

Shared work runs once per gameweek: ingest, then predict, which only *serves* the promoted
model. Training is a separate job on its own schedule. Per-user work (fetch squad, optimise)
is cheap and runs on demand.

| Component | Status |
|---|---|
| `common/` config, logging, Parquet I/O, FPL client | done |
| `ingest/` FPL API -> Parquet; `ingest.history` one-off import of past seasons | done |
| `ml/` features, model, registry, promotion gate, monitoring, backtest | done |
| `train/` train, evaluate, register and (if it passes the gate) promote a model | done |
| `predict/` serve the promoted model | done |
| `optimise/`, `notify/`, `api/` | not started |
| `.github/` CI and deploy | not started |

## Data layout

`gs://$DATA_BUCKET/season=<yyyy-yy>/gw=<n>/` (or `./data/...` locally):

- `players`, `teams`, `events`, `fixtures`: snapshot taken when ingest ran for gameweek `n`.
  The players snapshot holds FPL's `ep_next`, which cannot be recovered later and is the
  baseline for backtests.
- `live`: per-player stats for gameweek `n` once it has finished.
- `predictions`: per player for gameweek `n`: `xpts` (what the optimiser maximises), `ep_next`
  (a benchmark only), `availability`, `fix_n`, `model_version`.

Don't use `GAMEWEEK=<past>` to re-snapshot an old gameweek: it would store *today's* players
data under that gameweek.

Past seasons (2022-23 onwards) are imported once from the community archive
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) with
`uv run python -m ingest.history`. Its `xP` becomes `ep_next` in those snapshots, stored against the *following* gameweek because the
archive captured it after each gameweek's matches; gameweeks where
the archive recorded all zeros are stored as null. Plan: Parquet in GCS stays the source of
truth, with BigQuery external tables over it later (needs an infra PR).

## Model lifecycle

Training and serving are separate jobs with separate lifecycles:

- **`train`** builds the training set from every stored season, runs a leak audit on the stored
  `ep_next`, scores the last N finished gameweeks walk-forward (each by a model trained only on
  earlier data), fits the final model, and writes it to `models/<name>/<version>/` with a
  **model card** (data hash, features and dtypes, hyperparameters, seeds, library versions,
  git SHA, holdout metrics). Versions are immutable.
- **Promotion gate** (`ml/gate.py`): the new version only becomes `latest` if the leak audit is
  clean, it beats the rolling-form baseline and FPL's `ep_next` on the holdout, and it is not
  worse than the serving model on gameweeks out-of-sample for both. Every check, pass or fail,
  is recorded in the card; a rejected model is kept for inspection but never served.
- **`predict`** loads `latest`, builds features for the current gameweek only, **checks the
  feature schema against the card** (it stops rather than score with misaligned columns), and
  scores. It never trains, runs in seconds, and stamps `model_version` on every prediction. With
  no promoted model it stops with a clear error: run `train` first.
- **Monitoring** (`ml/monitor.py`, run by `train`): compares the predictions actually served
  with results and with `ep_next`, gameweek by gameweek, into
  `monitoring/season=<s>/live_performance.parquet`, and warns if the served model has lately
  trailed FPL's own figure.
- Reproducible: seeds and deterministic LightGBM, plus a data fingerprint, so the same data
  retrains to the same predictions.

Bucket layout additions: `models/<name>/<version>/...`, `models/<name>/latest.json` and
`monitoring/season=<s>/...`, alongside `season=<s>/gw=<n>/...`.

## Expected points

Expected points always come from the model: chance of playing x points if playing (LightGBM),
from player form and per-90 rates, team and opponent form, fixtures, rest days and last
season's rates, scaled by FPL's injury news. FPL's own `ep_next` is kept in the predictions only
as a benchmark for the backtest and live monitoring; it is never used as the expected points.

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
uv run python -m train.main               # train, evaluate, register, maybe promote a model
uv run python -m predict.main             # score the latest snapshot with the promoted model
uv run python -m ml.backtest              # regenerate docs/backtest.md (--no-ablation: lighter)
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
come (`uv run python -m ml.backtest`, or `--no-ablation` for the lighter version).

- `ep_next` is just recent form scaled by availability, with no fixture awareness.
- The archive's raw `xP` was captured *after* each gameweek and contains the outcome. The
  importer stores it against the following gameweek, and the backtest re-runs a leak audit
  every time. An earlier claim that FPL's `xP` beat the model was this leak.

`archive/` holds the original single-file optimiser, kept for reference only.
