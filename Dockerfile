# syntax=docker/dockerfile:1

# Multi-stage build for the optimus Discord moderation bot.
#
# A single image runs any of the six services; pick one at runtime via the
# container command, e.g. `python -m optimus.services.gateway`. The Python
# version is pinned to match pyproject's `requires-python = ">=3.12"` and CI
# (3.12); dependencies are installed from the committed `uv.lock` with
# `--frozen` so the image is fully reproducible.

# Runtime uses the full patch tag (published on Docker Hub); the uv builder
# image only publishes minor-version python tags, so it is pinned separately.
ARG PYTHON_VERSION=3.12.8
ARG UV_PYTHON_TAG=python3.12

# ---- Builder: resolve and install the locked dependency set -----------------
FROM ghcr.io/astral-sh/uv:0.9.9-${UV_PYTHON_TAG}-bookworm-slim AS builder

# Bytecode-compile on install and copy (not link) packages into the venv so the
# result is self-contained and relocatable into the runtime stage.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1. Install only third-party dependencies first, in a cached layer that is
#    invalidated only when the lockfile or project metadata changes. The source
#    tree is deliberately excluded here so editing code does not bust the cache.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 2. Install the project itself (its own package) on top of the cached deps.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- Runtime: slim final image with only what is needed to run --------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# curl is used by the compose healthchecks to probe /readyz; tini reaps the
# decode subprocesses the detection worker spawns so they cannot become zombies.
# Versions are intentionally unpinned so apt pulls current security patches.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl tini \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user; the app lives under /app owned by that user.
RUN groupadd --gid 10001 optimus \
    && useradd --uid 10001 --gid optimus --no-create-home --home-dir /app optimus

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the prebuilt virtualenv and the application source from the builder. The
# venv's interpreter is what `sys.executable` resolves to, so the sandboxed
# decode subprocess (Pillow + numpy) runs against this same environment.
COPY --from=builder --chown=optimus:optimus /app/.venv /app/.venv
COPY --from=builder --chown=optimus:optimus /app/src /app/src
COPY --from=builder --chown=optimus:optimus /app/migrations /app/migrations
COPY --from=builder --chown=optimus:optimus /app/alembic.ini /app/alembic.ini
COPY --from=builder --chown=optimus:optimus /app/scripts /app/scripts
COPY --from=builder --chown=optimus:optimus /app/pyproject.toml /app/pyproject.toml

USER optimus

EXPOSE 8080

ENTRYPOINT ["tini", "--"]
# Default to the gateway; override per service in compose or `docker run`.
CMD ["python", "-m", "optimus.services.gateway"]
