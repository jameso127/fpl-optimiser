# Build with the repo root (fpl-backend/) as context:
#   docker build -f predict/Dockerfile .
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group predict --no-install-project

COPY common ./common
COPY ingest ./ingest
COPY predict ./predict

CMD ["uv", "run", "--no-sync", "python", "-m", "predict.main"]
