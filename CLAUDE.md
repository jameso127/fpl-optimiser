# FPL Optimiser: Backend

Python services for a demo project: pulls Fantasy Premier League (FPL) API data, predicts
expected points per player, optimises squads and transfers, sends recommendations on Telegram. A
frontend and API are deferred.

**Hard constraint: must stay cheap (target < £2/month on GCP).** Prefer serverless,
scale-to-zero. Never add an always-on resource.

## Related repos

- `fpl-optimiser-infra`: ALL GCP infrastructure (Terraform), Cloud Workflows, Scheduler, IAM.
- `fpl-optimiser-web`: React frontend (deferred; there is no API service yet).

This repo contains application code only. **Do not add Terraform or create GCP resources
here.** If a change needs a new bucket, job, secret, IAM role or API enabled, stop and tell me
it needs an infra PR first.

### Contract with the infra repo (names must match)

Provided to this repo as GitHub Actions variables / runtime env vars:
- `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WIF_PROVIDER`, `GCP_DEPLOYER_SA`
- `ARTIFACT_REGISTRY_REPO` (image repo path)
- `DATA_BUCKET` (GCS bucket for Parquet data), layout: `gs://$DATA_BUCKET/season=<yyyy-yy>/gw=<n>/...`, plus `models/<name>/<version>/...`
  (model registry) and `monitoring/season=<yyyy-yy>/...`
- Cloud Run jobs (pre-created by infra): `fpl-ingest`, `fpl-train`, `fpl-predict`, `fpl-optimise`,
  `fpl-notify`
If I ask for a job/service that doesn't exist yet, say so rather than assuming it does.

## Architecture

```
Scheduler (daily 10:00 UK) -> Workflows -> Cloud Run Jobs: ingest -> predict -> optimise -> notify
   first, ingest with SCHEDULE_ONLY=true writes schedule.json (is the deadline today?).
   Not a deadline day: stop. Deadline day: run the four jobs.
Scheduler (weekly) -> Cloud Run Job: train
                                   |
                           Cloud Storage (Parquet)
```

- Shared, expensive work runs once per gameweek (ingest + predict for all players).
- Per-user work (optimise, notify) is cheap and runs once per deadline day.
- Storage is Parquet in GCS (no BigQuery unless asked). Users (chat id, FPL team id, settings,
  declared transfers, invites) are in Firestore behind a repository port, with an in-memory
  implementation for tests and single-user mode. Notifications go out over the Telegram Bot API.

## Layout

```
/ingest     Cloud Run Job: FPL API -> Parquet
/ml         Shared ML library: features, model, registry, promotion gate, monitoring, backtest
/train      Cloud Run Job: train, evaluate, register and maybe promote a model (own schedule)
/predict    Cloud Run Job: serve the promoted model -> expected points (never trains)
/optimise   Optimiser library + Cloud Run Job (squad, transfers, hit penalties)
/notify     Cloud Run Job: format recommendations and send them on Telegram
/common     Shared package (schemas, GCS I/O, config, logging)
/docs       Architecture diagram, backtest results
/.github    Workflows
```

## Stack and conventions

- Python 3.12, `uv`. Type hints everywhere.
- `ruff` (lint + format), `mypy`, `pytest`.
- Optimiser: HiGHS via SciPy's `milp` (no extra dependency). Model: LightGBM or ridge. Keep it simple.
- Each deployable has its own `Dockerfile`; shared code is in `/common`, copied into images.
  Build with the repo root as context.
- Config via environment variables (pydantic-settings). No hardcoded project IDs, bucket
  names or URLs.
- Structured JSON logging (Cloud Logging friendly).
- Jobs must be idempotent: re-running a gameweek overwrites that gameweek's outputs.
- No dry-run mode (a decision, to keep the code simple): tests use fake sources in `tests/fakes.py`.

## Data and modelling rules

- Always benchmark against FPL's `ep_next` in a backtest. Report results honestly in `/docs`,
  including if the model doesn't beat the baseline.
- No data leakage: features for gameweek N use only data available before N's deadline.
- Be a good API citizen: cache FPL responses, descriptive User-Agent, exponential backoff,
  never hammer endpoints.

## CI/CD (GitHub Actions) is the ONLY way things get deployed

- **Never run `gcloud run deploy`, `gcloud run jobs update` or `docker push` from a local
  machine.**
- Auth uses Workload Identity Federation (no service account keys, no JSON keys in secrets).
- `ci.yml` on every PR: ruff, mypy, pytest, docker build (no push).
- `deploy.yml` on merge to `main`:
  1. Build only components whose paths changed (path filters; changes to `/common` rebuild all).
  2. Push images to Artifact Registry tagged with the git SHA. Never deploy `latest`.
  3. Update the existing Cloud Run jobs / service to the new image
     (`gcloud run jobs update --image ...`, `gcloud run deploy --image ...`).
- Pin actions to a version or SHA. Use least-privilege `permissions:` on each workflow.

## Secrets

- No secrets in the repo, ever. Runtime secrets come from Secret Manager (wired up by the
  infra repo). Local dev uses a gitignored `.env`; keep `.env.example` current.
- If you see a secret in a diff, stop and tell me.

## Commands

```
make setup      # install deps
make lint       # ruff + mypy
make test       # pytest
make pipeline   # ingest -> predict -> optimise locally against ./data
```
(TODO: keep in sync with the Makefile.)

## How to work with me

- For anything non-trivial, propose a short plan first and wait for my OK.
- Small, focused changes; conventional commits. Add or update tests with each change.
  Don't claim something works without running it.
- Don't add dependencies or paid GCP usage without flagging cost and reason.
- If a request conflicts with this file, point it out rather than silently doing it.
- Keep the README current: architecture diagram, backtest results, links to the other repos.