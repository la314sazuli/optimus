# Contributing to optimus

Thanks for helping improve optimus. This guide covers local setup, the exact
checks CI enforces, how to run the services and tests, and what a reviewable PR
looks like. Every command here is run the same way in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) — if it passes locally it
passes in CI.

## Prerequisites

- **Python 3.12+** (`requires-python = ">=3.12"`).
- **[uv](https://docs.astral.sh/uv/)** — the only supported package/venv manager.
  The lockfile (`uv.lock`) is authoritative; do not use `pip` or edit it by hand.
- For running services end-to-end (not needed for the test suite): reachable
  **PostgreSQL**, **Redis**, and **NATS (JetStream-enabled)** instances.

## Setup

```bash
uv sync --extra dev --frozen
```

`--frozen` installs exactly what `uv.lock` pins and fails if `pyproject.toml`
and the lock have drifted — this is what CI does, so always use it. This creates
`.venv/` with the runtime deps plus the `dev` tooling (pytest, ruff, mypy,
hypothesis, fakeredis, aiosqlite, pre-commit). Prefix commands with `uv run` to
use that environment.

Optional extras (not required for development or the test suite):

```bash
uv sync --extra dev --extra embedding --extra evidence --frozen
```

- `embedding` — ONNX runtime for optional embedding confirmation of ambiguous
  matches (`OPTIMUS_EMBEDDING_ENABLED`).
- `evidence` — `aioboto3` for optional S3/MinIO evidence storage
  (`OPTIMUS_EVIDENCE_ENABLED`).

> **Do not change dependencies casually.** Keep `uv sync --frozen` working. If a
> change genuinely needs a new dependency, update `pyproject.toml` and regenerate
> the lock with `uv lock`, and call it out explicitly in the PR.

## The checks CI runs

CI runs four gates, in this order. All four must be green to merge:

```bash
uv run ruff check .         # lint
uv run ruff format --check . # formatting (no rewrite; fails on drift)
uv run mypy                 # type-check (strict; checks src/optimus only)
uv run pytest --cov=src/optimus --cov-report=term-missing
```

Apply formatting locally with `uv run ruff format .` before committing.

Notes that bite people:

- **`uv run mypy` takes no path argument.** The target is fixed in
  `pyproject.toml` (`[tool.mypy] files = ["src/optimus"]`), and `strict = true`
  is on. Type the `src/` tree; tests are not type-checked.
- **Ruff runs both lint and format checks in CI.** `ruff check` lints and
  `ruff format --check` enforces formatting (it only reports; it never rewrites
  in CI). The tree is `ruff format`-clean, so run `uv run ruff format .` and
  commit the result before pushing. Selected lint rule families (see
  `[tool.ruff.lint]`): `E F I N UP B C4 SIM RUF ASYNC S` (security `S` is on for
  `src/`, relaxed for `tests/`). If a `# noqa` is genuinely needed, scope it to
  the specific rule and add a one-line reason.
- **Coverage** is reported but there is no hard `--cov-fail-under` gate; the
  established baseline is ~80% line coverage across ~420 tests. Don't regress it —
  add tests for new code paths.

## Pre-commit hooks

A [`.pre-commit-config.yaml`](.pre-commit-config.yaml) is provided. It runs the
**same** `ruff check`, `ruff format --check`, and `mypy` as CI (via local hooks
that shell out to `uv run`, so versions and config match exactly — no
pinned-mirror drift), plus gitleaks secret scanning and basic file hygiene
(EOF/whitespace/YAML/TOML/merge-conflict/large-file checks).

Install and run it with [uv's tool runner](https://docs.astral.sh/uv/) (no global
install needed):

```bash
uvx pre-commit install         # run hooks automatically on every git commit
uvx pre-commit run --all-files # run the whole suite on demand
```

If you `uv sync --extra dev`, `pre-commit` is also on your path under
`uv run pre-commit ...`. A clean `pre-commit run --all-files` means the CI
lint+type stage will pass.

## Running the services locally

optimus is six small, single-purpose services that talk over a versioned NATS
event bus. Each is a module entrypoint and runs in its own process. They need
PostgreSQL, Redis, and JetStream-enabled NATS reachable via the `OPTIMUS_*`
settings.

### Docker Compose (full stack)

The quickest way to get every service plus its datastores running:

```bash
cp .env.example .env          # set OPTIMUS_DISCORD_TOKEN
docker compose up --build     # postgres + redis + nats + migrate + 6 services
```

Compose starts PostgreSQL, Redis, and NATS (with JetStream), runs the `migrate`
one-shot (`alembic upgrade head`) once they are healthy, then starts the six
services — each gated on the datastores' healthchecks and the migration
completing. Container healthchecks hit each service's `/readyz`. Register slash
commands once with `docker compose run --rm gateway python
scripts/register_commands.py`. The image (see [`Dockerfile`](Dockerfile)) is a
single multi-stage build shared by all services; the service is chosen by the
container `command`.

### Manual (uv, separate processes)

```bash
# 1. Configure
cp .env.example .env
# edit .env: at minimum OPTIMUS_DISCORD_TOKEN and the datastore URLs

# 2. Apply database migrations
uv run alembic upgrade head

# 3. Register slash commands with Discord (once per command-set change)
uv run python scripts/register_commands.py

# 4. Run each service (separate processes/terminals)
uv run python -m optimus.services.gateway        # Discord connection, image extraction
uv run python -m optimus.services.ingest         # SSRF-safe fetch
uv run python -m optimus.services.detection      # decode + hash + match + swarm
uv run python -m optimus.services.moderation     # policy -> action, review channel
uv run python -m optimus.services.interactions   # slash commands + buttons
uv run python -m optimus.services.scheduler      # retention, rollups, index rebuild
```

Configuration is environment-driven (prefix `OPTIMUS_`); see
[`.env.example`](.env.example) and
[`src/optimus/core/config.py`](src/optimus/core/config.py) for every setting,
its default, and its validation bounds. The `gateway` and `interactions`
services need a valid `OPTIMUS_DISCORD_TOKEN`; the pure-backend services
(`ingest`, `detection`, `moderation`, `scheduler`) only need the datastore URLs.

Each service exposes a health server on `OPTIMUS_HEALTH_PORT` (default `8080`):
`/healthz` (liveness), `/readyz` (readiness — probes the service's NATS/Redis
dependencies and returns 503 while a backing store is unreachable), and
`/metrics` (Prometheus). See [`docs/architecture.md`](docs/architecture.md) for
how the services fit together and how a message flows through the pipeline.

## Tests

```bash
uv run pytest                       # whole suite (quiet, asyncio auto-mode)
uv run pytest tests/unit/test_ssrf.py        # one file
uv run pytest -k circuit            # by keyword
uv run pytest --cov=src/optimus --cov-report=term-missing   # with coverage
```

`pytest-asyncio` is in `asyncio_mode = "auto"`, so `async def test_*` functions
run without an explicit marker.

### Test layout & conventions

- **`tests/unit/`** — fast, isolated, no external services. This is the bulk of
  the suite and where most new tests belong.
- **`tests/integration/`** — wider wiring (fetcher, scheduler loop, index
  manager) that still runs in-process without real infra.
- **`tests/fixtures/`** — deterministic scam/clean image fixtures and
  `labels.json`, generated by `scripts/make_fixtures.py` and used by the
  detection-quality eval (`scripts/eval.py`).
- **No real infrastructure in tests.** Postgres is replaced by an in-memory
  **aiosqlite** engine (the shared `session` fixture in `tests/conftest.py`
  creates the full schema per test); Redis is replaced by **fakeredis**
  (`fakeredis.aioredis`). Do not require a live database, Redis, or NATS to run
  the suite.
- **Inject time; never sleep.** Time-dependent components
  (e.g. `CircuitBreaker`, rate limiters) take an injectable time source. Tests
  drive a fake clock — typically a mutable `clock = {"t": 0.0}` passed as
  `time_source=lambda: clock["t"]` and advanced explicitly — instead of real
  delays. Add the same hook to new time-dependent code so it stays testable and
  the suite stays fast and deterministic.
- **Property tests** use **hypothesis** (`@given(...)`) for algorithmic
  invariants — e.g. perceptual hashing, the BK-tree, SSRF range checks,
  import-schema validation, safe-mode baselines. When you add or change an
  algorithm with an invariant ("BK-tree query never misses a match within
  radius", "every private range is rejected"), prefer a property test alongside
  the example-based ones; see `tests/unit/test_properties.py`.

## Pull request expectations

- **All four CI gates green:** `ruff check .`, `ruff format --check .`, `mypy`
  (strict), and the full `pytest` suite. PRs that don't pass these will not be
  merged.
- **Don't regress coverage** (~80% baseline). New code paths need tests; new
  algorithmic invariants warrant a property test.
- **Keep `uv sync --frozen` working** — no accidental lockfile drift, no
  unjustified new dependencies.
- **Security-sensitive code** (SSRF guard, decode sandbox, permission re-checks,
  signing, Redis key/TTL discipline, rate limits) is held to a high bar.
  Preserve the fail-closed behavior and the invariants documented in
  [`docs/security-audit.md`](docs/security-audit.md); never log secrets.
- **Match the codebase.** Read nearby code first; follow existing module
  structure, naming, and the "small single-purpose service over NATS" shape.
  Comments explain *why*, not *what*.
- **Scope tightly.** One logical change per PR. Describe what changed and why,
  and how you verified it (the commands above).

## License

By contributing you agree your contributions are licensed under the project's
[MIT License](LICENSE).
