# optimus

[![CI](https://github.com/la314sazuli/optimus/actions/workflows/ci.yml/badge.svg)](https://github.com/la314sazuli/optimus/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Open-source Discord moderation bot focused on detecting and removing scam,
phishing, and fraud **images** (fake giveaways, fake Nitro/Steam gifts, fake
exchange screenshots, wallet-drainer QR codes) in near-real-time, built to scale
to very large guilds.

optimus matches uploaded images against a database of known scam-campaign
images using a four-hash **perceptual hashing** ensemble (pHash, dHash, wHash,
aHash). It is resilient to the re-share transforms scammers use — cropping,
re-coloring, re-compression, resizing, and watermarking — while keeping a
zero-false-positive bias so auto-moderation never punishes legitimate users.

## How it works

Images flow through a set of small, single-purpose services that communicate
over a versioned [NATS](https://nats.io) event bus. State lives in PostgreSQL
(per-guild config, hashes, detections, audit log) and Redis (rate limits,
idempotency, swarm windows, safe-mode baselines).

```
Discord ──▶ gateway ──▶ ingest ──▶ detection ──▶ moderation ──▶ Discord
              │            │           │              │
              │   (SSRF-safe fetch)    │      (delete/timeout/ban,
              │            │           │       mod-review channel)
              │            │           │
              └── slash commands / buttons ──▶ interactions
                                                  │
                                       scheduler (retention, rollups,
                                       evidence GC, index rebuild)
```

| Service        | Responsibility |
| -------------- | -------------- |
| `gateway`      | Connects to Discord, extracts image attachments/links, emits `message_image` events. |
| `ingest`       | Fetches images through an **SSRF-hardened** fetcher (DNS pinning, size caps, magic-byte sniffing) and emits `image_fetched`. |
| `detection`    | Decodes images under CPU/memory/pixel limits, computes the hash ensemble, matches against guild + global indexes (BK-tree), correlates cross-guild **swarms**, and emits a `verdict`. |
| `moderation`   | Maps a verdict + guild policy onto an action (report / delete / timeout / ban), enforces per-guild rate limits and circuit breakers, posts to the mod-review channel, and can flip a guild into **safe mode** on anomalous spikes. |
| `interactions` | Handles slash commands and review buttons (config, hash add/import/export, appeals) with server-side permission re-checks. |
| `scheduler`    | Periodic jobs: data retention, metric rollups, evidence GC, and hash-index rebuilds. |

Event schemas and NATS subjects are defined in
[`src/optimus/contracts/events.py`](src/optimus/contracts/events.py); every
subject is versioned (`...v1`) so schemas can evolve without breaking consumers.

For a deeper treatment — the full message flow, the detection pipeline, where
state lives, and where the resilience controls sit (with diagrams) — see
[`docs/architecture.md`](docs/architecture.md).

## Quickstart (Docker Compose)

The fastest way to self-host: one image runs every service, and Compose brings
up the backing stores (PostgreSQL, Redis, JetStream-enabled NATS), applies
migrations, and starts the six services. Requires Docker with the Compose plugin.

```bash
# 1. Configure
cp .env.example .env
# edit .env: set OPTIMUS_DISCORD_TOKEN (and POSTGRES_PASSWORD for non-local use)

# 2. Build the image and start the whole stack
docker compose up --build
```

Compose runs the `migrate` one-shot (`alembic upgrade head`) before the services
start, and gates each service on the datastores reporting healthy. The services'
`/readyz` probes drive the container healthchecks. Slash commands still need a
one-time registration:

```bash
docker compose run --rm gateway python scripts/register_commands.py
```

The image is multi-stage (uv installs the locked, `--frozen` dependency set; the
final stage is a slim, non-root `python:3.12-slim` with only runtime deps) and
service-agnostic — select a service via its command, e.g.
`python -m optimus.services.detection`.

## Quickstart (self-hosted, no Docker)

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and reachable
PostgreSQL, Redis, and NATS instances.

```bash
# 1. Install dependencies (and dev tooling)
uv sync --extra dev

# 2. Configure
cp .env.example .env
# edit .env: set OPTIMUS_DISCORD_TOKEN and the datastore URLs

# 3. Apply database migrations
uv run alembic upgrade head

# 4. Register slash commands with Discord (once per command change)
uv run python scripts/register_commands.py

# 5. Run the services (each in its own process)
uv run python -m optimus.services.gateway
uv run python -m optimus.services.ingest
uv run python -m optimus.services.detection
uv run python -m optimus.services.moderation
uv run python -m optimus.services.interactions
uv run python -m optimus.services.scheduler
```

Each service exposes a health endpoint (`/healthz`, `/readyz`) on
`OPTIMUS_HEALTH_PORT` and Prometheus metrics for observability. Readiness
probes the service's NATS, Redis, and (for interactions) Postgres dependencies,
so `/readyz` returns 503 while a backing store is unreachable; each probe is
bounded by a timeout so a black-holed dependency fails closed.

## Configuration

All settings are read from the environment with the `OPTIMUS_` prefix (or a
`.env` file); see [`.env.example`](.env.example) for the full list and
[`src/optimus/core/config.py`](src/optimus/core/config.py) for defaults and
validation bounds. Highlights:

| Variable | Purpose |
| -------- | ------- |
| `OPTIMUS_TENANCY` | `single` (self-hosted) or `multi` (SaaS, row-level-security multi-tenant). |
| `OPTIMUS_DISCORD_TOKEN` | Bot token. |
| `OPTIMUS_DATABASE_URL` / `OPTIMUS_REDIS_URL` / `OPTIMUS_NATS_URL` | Datastore connections. |
| `OPTIMUS_SENSITIVITY_DEFAULT` | `strict` / `balanced` / `permissive` matching preset (per-guild overridable). |
| `OPTIMUS_INGEST_MAX_BYTES` | Hard cap on fetched image size (streamed, never fully buffered if exceeded). |
| `OPTIMUS_EMBEDDING_ENABLED` | Optional ONNX embedding confirmation for ambiguous matches (`embedding` extra). |
| `OPTIMUS_EVIDENCE_ENABLED` | Optional S3/MinIO evidence storage with presigned, TTL'd URLs (`evidence` extra). |
| `OPTIMUS_GLOBAL_SIGNING_PUBLIC_KEY` / `..._PRIVATE_KEY` | Ed25519 keys for the signed global hash database. |

Per-guild settings (sensitivity, action policy, thresholds, locale, opt-ins)
are stored in the database and changed at runtime via `/config set`.

## Detection quality

The ensemble weights and per-preset thresholds are tuned against a deterministic
fixture set (`scripts/make_fixtures.py`) and evaluated by `scripts/eval.py`. The
current baseline is documented in [`docs/eval/baseline.md`](docs/eval/baseline.md):
precision and false-positive rate are **perfect** across all presets, with recall
trading off per sensitivity — the right bias for an auto-moderation action.

For a deeper offline evaluation — a synthetic corpus run through the real
pipeline across a full threshold sweep, with per-perturbation recall and a
recommended operating point — run `python -m benchmarks`. See
[`docs/detection-eval.md`](docs/detection-eval.md) for usage and findings.

## Security model

- **SSRF defense.** Untrusted image URLs are validated by
  [`optimus.ingest.ssrf`](src/optimus/ingest/ssrf.py): DNS is resolved once and
  the IP is *pinned* for the connection (closing DNS-rebinding), every resolved
  address is checked against private/loopback/link-local/CGNAT/reserved/metadata
  ranges (IPv4 and IPv6, including IPv4-mapped addresses), non-Discord hosts must
  use HTTPS, and each redirect hop is re-validated.
- **Decode sandboxing.** Image decoding runs under CPU-time, memory, pixel, and
  frame limits to bound the cost of decompression-bomb inputs.
- **Defense-in-depth permissions.** A component's `default_member_permissions`
  is treated only as a client hint; every state-changing interaction is
  re-checked server-side against the invoker's effective permissions.
- **Signed global database.** Promoted global hashes are Ed25519-signed over a
  canonical encoding; consumers reject any record that fails verification.
- **Multi-tenant isolation.** In `multi` tenancy, PostgreSQL row-level security
  (see [`migrations/`](migrations)) isolates each guild's data.
- **Idempotency & abuse control.** Per-attachment idempotency keys prevent
  double-acting on retries; Redis token buckets and circuit breakers bound the
  Discord REST action rate per guild.

## Development

```bash
uv sync --extra dev --frozen
uv run ruff check .   # lint (matches CI)
uv run mypy           # type-check, strict (matches CI)
uv run pytest         # test suite (matches CI)
```

Optional [pre-commit](https://pre-commit.com/) hooks run the same `ruff check`
and `mypy` plus secret scanning:

```bash
uvx pre-commit install
uvx pre-commit run --all-files
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full developer workflow (setup,
running services locally, test conventions, and PR expectations) and
[`docs/architecture.md`](docs/architecture.md) for the system design. Additional
references: [`docs/security-audit.md`](docs/security-audit.md),
[`docs/performance-notes.md`](docs/performance-notes.md), and
[`docs/eval/baseline.md`](docs/eval/baseline.md).

## License

MIT — see [LICENSE](LICENSE).
