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
#    Plain COPY (not --mount=type=bind) is used because Railway's builder only
#    accepts type=cache mounts on RUN; other mount types are rejected outright.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,id=s/df3ac583-26fc-4f39-98d7-90e26a2e4474-/root/.cache/uv,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2. Install the project itself (its own package) on top of the cached deps.
COPY README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY scripts ./scripts
RUN --mount=type=cache,id=s/df3ac583-26fc-4f39-98d7-90e26a2e4474-/root/.cache/uv,target=/root/.cache/uv \
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
    PATH="/app/.venv/bin:$PATH" \
    OPTIMUS_SIMPLE_DATABASE_URL="sqlite+aiosqlite:////data/optimus.db"

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

# Simple mode persists its SQLite database under /data, a directory kept separate
# from the read-only app install at /app. Attach persistent storage at this path
# via your platform's volume mechanism (e.g. `docker run -v optimus-data:/data`,
# or a Railway Volume mounted at /data in the service settings) -- this image
# deliberately does not declare a Docker VOLUME, since Railway's builder rejects
# that instruction and some platforms treat it as an unmanaged anonymous volume
# anyway. /data is owned by the unprivileged runtime user so the first boot can
# create the database file (a volume mounted over /app would be root-owned and
# unwritable). OPTIMUS_SIMPLE_DATABASE_URL above points the engine at /data.
RUN mkdir -p /data && chown optimus:optimus /data

USER optimus

EXPOSE 8080

ENTRYPOINT ["tini", "--"]
# Default to simple mode: the whole bot in one process (OPTIMUS_MODE=simple).
# A single `docker run -e OPTIMUS_DISCORD_TOKEN=... optimus` runs the bot with no
# external services. Distributed deployments override the command per service
# (e.g. `python -m optimus.services.gateway`), as docker-compose.yml does.
CMD ["python", "-m", "optimus"]
