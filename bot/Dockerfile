# Build with the repo root (fpl-backend/) as context:
#   docker build -f bot/Dockerfile .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# A Cloud Run service (the Telegram webhook), not a job: it reads and writes users and calls the
# FPL and Telegram APIs. No ML, solver or drawing libraries.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group firestore --no-install-project

COPY common ./common
COPY bot ./bot

CMD ["uv", "run", "--no-sync", "python", "-m", "bot.main"]
