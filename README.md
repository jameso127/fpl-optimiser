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
| `optimise/` squad and transfer optimiser (MILP, HiGHS via SciPy) | done |
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

## Users and pending transfers

People use the bot through Telegram. `common/users/` holds who they are: a chat id, a public
FPL team id, a few settings. There are no names, emails or credentials, and `/delete` removes
everything.

- **Storage is behind a port** (`UserRepository`), with a Firestore implementation for
  production and an in-memory one for tests and local runs. One contract test suite runs
  against both (the Firestore leg uses the emulator in CI), so they cannot drift apart.
  Operations that must not race (redeeming an invite, claiming a notification so a gameweek is
  never sent twice) are single atomic methods, tested with concurrent callers.
- **Registration is by invite code**, so strangers cannot add load on the FPL API or on costs.
  Declared transfers and invites expire automatically (Firestore TTL).
- **FPL hides pending changes.** Before a gameweek's deadline, the picks endpoint for it returns
  404, and the entry summary only shows the last deadline's bank, value and transfer count.
  So a manager's transfers made since the last deadline are invisible to us. Users therefore
  tell the bot about transfers they have already made; they are applied on top of the last
  visible squad with the prices recorded at that moment (cash, selling price, and free
  transfers all follow). A declared transfer that cannot be true (already visible, selling a
  player not owned, unaffordable) is ignored with a note instead of corrupting the squad.
- **Every recommendation lists its assumptions** ("your team as FPL shows it after the
  gameweek 5 deadline", "plus the transfers you told me about: ...", "free transfers left: 2
  (estimated)") so a wrong assumption is visible to the reader.

## Optimiser

`optimise` turns the predictions into advice for one manager (`FPL_TEAM_ID`). For each number
of transfers 0..N (`MAX_TRANSFERS`, default 4) it solves a mixed-integer program (SciPy's
`milp`, which uses the HiGHS solver) for the best squad, starting XI, captain, vice-captain
and bench order, and writes `season=<s>/gw=<n>/recommendations/<team_id>.json`.

- **Formulation:** binary variables per player for squad, starter and captain. Maximise the
  starters' expected points plus the captain's second share and a small bench weight. Subject
  to 15 players (2/5/5/3), at most 3 per club, the budget, a legal XI (1 GK, 3-5 DEF, 2-5 MID,
  1-3 FWD), one captain who starts, and exactly k players changed. The hit is
  `4 x max(0, k - free transfers)`.
- **Budget** is the squad's *selling* prices plus the bank; players already owned cost their
  selling price, everyone else the market price.
- **Recommendation:** the best net option, but only if each transfer earns at least
  `MIN_GAIN_PER_TRANSFER` (0.5) expected points over holding, preferring fewer transfers
  within 0.25 points. Predictions are noisy, so small projected gains are not acted on.
- **Squad reconstruction** (`optimise/squad.py`) is the fragile part: FPL does not expose
  selling prices or free transfers, so they are derived from the picks, transfer history and
  chips (Free Hit weeks handled). Free transfers are an estimate; set
  `FREE_TRANSFERS_OVERRIDE` when it is wrong (FPL occasionally grants top-ups the API does not
  show).
- **Limits:** one gameweek ahead only (the model predicts one gameweek), and chips are not
  modelled.

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
uv run python -m ingest.main              # live FPL API -> ./data
uv run python -m ingest.history           # one-off: past seasons -> ./data
uv run python -m train.main               # train, evaluate, register, maybe promote a model
uv run python -m predict.main             # score the latest snapshot with the promoted model
uv run python -m optimise.main            # best team for 0..N transfers (needs FPL_TEAM_ID)
uv run python -m ml.backtest              # regenerate docs/backtest.md (--no-ablation: lighter)
```

Copy `.env.example` to `.env` for local settings. No secrets are committed.

## Repos

- `fpl-optimiser-infra`: all GCP infrastructure (Terraform, Workflows, Scheduler, IAM).
- `fpl-optimiser-web`: React frontend.

## Backtest results

Full report: [docs/backtest.md](docs/backtest.md), regenerated by `uv run python -m ml.backtest`
(walk-forward over all 36 testable gameweeks of 2025-26 plus this season; each gameweek is
scored by a model trained only on earlier data; hyperparameters were tuned on 2024-25 only).

- **Beats the simple baselines clearly.** 2025-26, all players: top-30 picks average 4.45
  points against 3.76 for a rolling-form baseline; regular-player Spearman 0.29 vs 0.14.
- **Beats FPL's `ep_next`** on the 9 gameweeks that have a clean pre-deadline value: top-30
  picks 4.52 vs 3.93 points, regular-player Spearman 0.30 vs 0.19, MAE 2.29 vs 2.67.
  `ep_next` is recent form scaled by availability, with no fixture awareness.
- **Beats the first model, but only slightly:** top-30 4.45 vs 4.40 and regular Spearman 0.29
  vs 0.28; regular-player MAE is marginally worse (2.35 vs 2.31).
- **Most of the new features do not measurably earn their place.** The ablation (12 test
  gameweeks, so noisy) shows only the player-detail group helping (regular Spearman 0.31 vs
  0.28 without it). Team form, rest days, and last season's rates change nothing measurable,
  and a hurdle model on the first model's features is about as good on top-30 points. They are
  candidates to prune once more gameweeks give a clearer reading.
- Beware the archive's raw `xP`: it was captured *after* each gameweek and contains the
  outcome. The importer stores it against the following gameweek, and the backtest re-runs a
  leak audit every time. An earlier claim that FPL's `xP` beat the model was this leak.

`archive/` holds the original single-file optimiser, kept for reference only.
