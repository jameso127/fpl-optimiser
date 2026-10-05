# Build with the repo root (fpl-backend/) as context:
#   docker build -f notify/Dockerfile .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# No ML or solver libraries: it only reads JSON, sends messages and (optionally) reads users.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group firestore --no-install-project

COPY common ./common
COPY notify ./notify

CMD ["uv", "run", "--no-sync", "python", "-m", "notify.main"]
