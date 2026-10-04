# Build with the repo root (fpl-backend/) as context:
#   docker build -f ingest/Dockerfile .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY common ./common
COPY ingest ./ingest

CMD ["uv", "run", "--no-sync", "python", "-m", "ingest.main"]
