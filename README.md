# FPL Optimiser

Predicts Fantasy Premier League points with a machine-learning model, finds the best squad and
transfers for a manager with an optimiser, and sends the advice to Telegram before each
deadline. It runs as small serverless jobs on Google Cloud and is designed to cost under £2 a
month: everything scales to zero.

The infrastructure is in a separate repo, `fpl-optimiser-infra` (Terraform).

## How it works

```
Scheduler -> Workflows -> Cloud Run jobs:  ingest -> predict -> optimise -> notify
Scheduler -> Cloud Run job:                train   (weekly, separate from the above)
                                |
              Cloud Storage: Parquet data, model registry, recommendations
```

| Job | What it does |
|---|---|
| `ingest` | Saves a snapshot of the FPL API (players, teams, fixtures, live stats) as Parquet. |
| `train` | Trains and evaluates a model, registers it, and only promotes it if it passes a gate. |
| `predict` | Scores every player for the next gameweek with the promoted model. Never trains. |
| `optimise` | For each user, finds the best team for 0 to N transfers, counting the points hit. |
| `notify` | Sends each user their recommendation on Telegram, once per gameweek. |

## Decisions worth a look

**Training and serving are separate.** `train` writes an immutable, versioned model with a
model card (data hash, features, parameters, library versions, git SHA, evidence). `predict`
only loads the promoted version, and refuses to score if the features no longer match what the
model was trained on.

**A promotion gate.** A new model only replaces the serving one if the leak audit is clean, it
beats a rolling-form baseline and FPL's own `ep_next`, and it is no worse than the current
model. A rejected model is kept, with the reasons, but never served.

**Honest backtesting.** Walk-forward over past seasons, every gameweek scored by a model that
only saw earlier data. An early version appeared to show FPL's expected points beating the
model; that turned out to be a leak (the archive's value contains the gameweek's own result).
The backtest now audits for it on every run. See [docs/backtest.md](docs/backtest.md).

**Optimiser as a mixed-integer program.** SciPy's `milp` (HiGHS): 15 players (2/5/5/3), at most
3 per club, budget on selling prices, a legal XI, a captain, and exactly *k* transfers. Each
extra transfer must earn at least 0.5 expected points to be recommended, because predictions
are noisy.

**FPL hides pending transfers.** Before a deadline, a manager's changes for that gameweek
cannot be read from the API. Users tell the bot about transfers they have made; they are
applied on top of the last visible squad, and impossible ones are ignored with a note. Every
recommendation lists the assumptions it is based on.

**Users are behind a port.** `UserRepository` has a Firestore implementation and an in-memory
one, and one contract test suite runs against both. Redeeming an invite and claiming a
notification are atomic, so a gameweek is never sent twice.

## Backtest

Walk-forward over 2025-26 and this season ([full report](docs/backtest.md)):

- Beats a rolling-form baseline: top-30 picks average 4.45 points against 3.76.
- Beats FPL's `ep_next` on the 9 gameweeks with a clean pre-deadline value: 4.52 against 3.93
  points, and a better rank correlation among regular players (0.30 against 0.19).
- Barely beats my first model (4.45 against 4.40), and most of the added features do not earn
  their place yet. They are candidates to prune once there is more data.

## Run it locally

Needs [uv](https://docs.astral.sh/uv/). Copy `.env.example` to `.env` and fill in your FPL team
id, a Telegram bot token (from @BotFather) and your chat id.

```
uv sync
uv run python -m ingest.history   # once: past seasons, for training
uv run python -m ingest.main      # the live FPL API -> ./data
uv run python -m train.main       # train, evaluate, register, maybe promote
uv run python -m predict.main
uv run python -m optimise.main
uv run python -m notify.main      # sends the message
```

`make pipeline` runs the last five. `make lint` and `make test` run ruff, mypy and pytest.

## Layout

```
common/    settings, logging, Parquet storage, FPL client, users store
ingest/    FPL API -> Parquet, plus the one-off history import
ml/        features, model, registry, promotion gate, monitoring, backtest
train/     predict/     optimise/     notify/     one Cloud Run job each, each with a Dockerfile
tests/     fakes and unit tests (the Firestore contract tests need the emulator)
```

## Status

- Working and tested locally: all five jobs, the optimiser and the user store.
- Written but not run yet: the CI and deploy workflows, the Dockerfiles and the Firestore
  implementation. They are verified by the first pull request.
- Not built yet: the interactive Telegram bot (registration by invite, `/xi`, declaring
  transfers) and a frontend.

`archive/` holds the original single-file optimiser, kept for reference only.
