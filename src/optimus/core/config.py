"""Application configuration loaded from the environment (prefix ``OPTIMUS_``)."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from optimus.globaldb.promotion import MIN_DISTINCT_APPROVERS


def _parse_shard_ids(raw: str) -> tuple[int, ...]:
    """Parse a shard-id spec into a sorted, de-duplicated tuple.

    Accepts comma-separated ids and inclusive ranges, e.g. ``"0,1,2"`` or
    ``"0-3"`` or ``"0-1,4,6-7"``. Whitespace is ignored. Ids are non-negative
    (a leading ``-`` is read as a range separator); an inverted range
    (``"3-1"``) raises :class:`ValueError`.
    """
    ids: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo_str, _, hi_str = token.partition("-")
            lo, hi = int(lo_str.strip()), int(hi_str.strip())
            if lo > hi:
                raise ValueError(f"inverted shard-id range: {token!r}")
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(token))
    return tuple(sorted(ids))


def _parse_key_map(raw: str) -> dict[str, str]:
    """Parse a ``"key_id:b64,key_id2:b64"`` spec into an id -> public-key map.

    Whitespace and blank entries are ignored. A malformed entry (missing ``:``
    or a duplicate id) raises :class:`ValueError` so a bad rotation config fails
    loudly at startup rather than silently dropping a key.
    """
    keys: dict[str, str] = {}
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        key_id, sep, value = token.partition(":")
        key_id, value = key_id.strip(), value.strip()
        if not sep or not key_id or not value:
            raise ValueError(f"malformed signing-key entry: {token!r} (expected 'key_id:b64')")
        if key_id in keys:
            raise ValueError(f"duplicate signing key_id: {key_id!r}")
        keys[key_id] = value
    return keys


def _parse_int_csv(raw: str) -> frozenset[int]:
    """Parse a comma-separated list of ints into a frozenset (blanks ignored)."""
    return frozenset(int(t.strip()) for t in raw.split(",") if t.strip())


def _parse_str_csv(raw: str) -> frozenset[str]:
    """Parse a comma-separated list of strings into a frozenset (blanks ignored)."""
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


class Mode(StrEnum):
    """Process-composition mode.

    ``SIMPLE`` (the default) runs every service in one asyncio process over an
    in-process bus with SQLite and in-memory backends — zero external services,
    ideal for getting started. ``DISTRIBUTED`` is the six-service deployment wired
    over NATS/Redis/Postgres, run via the per-service entrypoints.
    """

    SIMPLE = "simple"
    DISTRIBUTED = "distributed"


class Tenancy(StrEnum):
    """Deployment tenancy mode."""

    SINGLE = "single"
    MULTI = "multi"


class Sensitivity(StrEnum):
    """Per-guild detection sensitivity preset."""

    STRICT = "strict"
    BALANCED = "balanced"
    PERMISSIVE = "permissive"


class RateLimitBackend(StrEnum):
    """Rate-limiter backend selection.

    ``MEMORY`` keeps the legacy per-process token bucket (correct for a single
    replica). ``REDIS`` shares one bucket across replicas via Redis so effective
    limits do not multiply with replica count.
    """

    MEMORY = "memory"
    REDIS = "redis"


class Settings(BaseSettings):
    """Runtime settings.

    Every field maps to an ``OPTIMUS_``-prefixed environment variable.
    """

    model_config = SettingsConfigDict(
        env_prefix="OPTIMUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: Process-composition mode. ``simple`` (default) runs everything in one
    #: process with zero external services; ``distributed`` is the six-service
    #: NATS/Redis/Postgres deployment. See :class:`Mode`.
    mode: Mode = Mode.SIMPLE

    tenancy: Tenancy = Tenancy.SINGLE

    # Discord
    discord_token: str = ""
    discord_client_id: str = ""
    discord_client_secret: str = ""

    # Gateway sharding
    #: Total number of gateway shards across the whole deployment. Unset (None)
    #: defers to hikari's automatic sharding (recommended for small fleets);
    #: Discord mandates sharding past ~2,500 guilds. When set, every replica
    #: must agree on the same value.
    shard_count: int | None = Field(default=None, ge=1)
    #: Which shard ids THIS replica should run, as a spec string parsed into a
    #: sorted tuple of ids (e.g. ``"0,1"`` or ``"0-3"``). Unset (None) means
    #: this replica runs all shards (the single-process default). When set,
    #: ``shard_count`` must also be set and every id must be ``< shard_count``.
    shard_ids: tuple[int, ...] | None = None

    # Datastores
    database_url: str = "postgresql+asyncpg://optimus:optimus@localhost:5432/optimus"
    redis_url: str = "redis://localhost:6379/0"
    #: Per-operation socket timeout (seconds) for the async Redis client. Bounds
    #: how long a single command blocks when Redis is unreachable so a dead Redis
    #: surfaces as a fast error the caller can degrade on (rate-limit/idempotency
    #: fallbacks) instead of stalling the event loop on the default unbounded
    #: wait. Also used as the connect timeout.
    redis_socket_timeout: float = Field(default=2.0, gt=0.0)
    nats_url: str = "nats://localhost:4222"
    #: SQLite file used in ``simple`` mode (zero external services). The async
    #: driver (aiosqlite) is already a dependency; nothing else is required.
    #: Override with ``OPTIMUS_SIMPLE_DATABASE_URL`` to point at a different path.
    simple_database_url: str = "sqlite+aiosqlite:///optimus.db"
    #: How long a SQLite writer waits for a competing lock before raising
    #: ``database is locked``. Simple mode runs the whole pipeline (ingest,
    #: detection, moderation) plus interaction handlers as concurrent writers
    #: against one file; SQLite serialises writers, so without a generous busy
    #: timeout a writer that collides with another in-flight write fails
    #: immediately. WAL (set alongside this) lets readers proceed during a write;
    #: the timeout absorbs writer-vs-writer overlaps under sustained load. 30s is
    #: well above the longest observed write burst while still bounding a genuine
    #: deadlock.
    sqlite_busy_timeout_ms: int = Field(default=30_000, ge=0)

    # Database connection pool (SQLAlchemy async engine, QueuePool).
    #: Persistent connections kept open per engine (i.e. per process/replica).
    #: At large fleets the real cap is ``replicas * (pool_size + max_overflow)``
    #: against one Postgres ``max_connections``; size accordingly or front the
    #: DB with an external pooler (see docs/operations.md).
    db_pool_size: int = Field(default=5, ge=1)
    #: Extra connections opened past ``pool_size`` under burst, closed when idle.
    db_max_overflow: int = Field(default=10, ge=0)
    #: Recycle (reconnect) a pooled connection after this many seconds. Guards
    #: against server-side/idle-proxy timeouts severing long-lived connections.
    db_pool_recycle: int = Field(default=1800, ge=1)
    #: Check a connection's liveness with a lightweight ping before handing it
    #: out, transparently replacing one the server has dropped.
    db_pool_pre_ping: bool = True

    # Logging / observability
    log_level: str = "INFO"
    service_name: str = "optimus"
    otel_enabled: bool = False
    otel_endpoint: str = ""

    # Health server
    health_host: str = "0.0.0.0"  # noqa: S104 - intended bind for containerized service
    health_port: int = 8080

    # Ingest
    ingest_max_bytes: int = 10 * 1024 * 1024
    #: Hard cap on the raw image size shipped inline (base64) through NATS in an
    #: ``image_fetched`` payload. A fetch may stream up to ``ingest_max_bytes``,
    #: but anything larger than this is resolved (dropped + metric) rather than
    #: ballooning the JetStream stream and detection-replica memory under a raid.
    #: Defaults below ``ingest_max_bytes`` so the inline path stays bounded even
    #: if the download cap is later raised; must not exceed ``ingest_max_bytes``.
    ingest_max_inline_bytes: int = 8 * 1024 * 1024
    ingest_max_redirects: int = 3
    #: Max inspectable images processed per Discord message. Beyond this the
    #: extras are dropped (counted) so one message with a flood of attachments
    #: cannot fan out an unbounded number of fetch/decode jobs.
    gateway_max_attachments: int = Field(default=10, ge=1)
    ingest_fetch_rate_capacity: float = 20.0
    ingest_fetch_rate_refill: float = 10.0
    #: Opportunistic idle-bucket sweep cadence for the in-memory rate-limiter
    #: fallback (seconds), used only when Redis is unavailable.
    ingest_inmemory_sweep_seconds: float = Field(default=300.0, gt=0.0)

    # Swarm correlation
    swarm_min_guilds: int = 3
    swarm_window_seconds: int = 300
    swarm_match_radius: int = 6

    # Detection
    decode_timeout_seconds: float = 5.0
    decode_cpu_seconds: int = 5
    decode_mem_bytes: int = 512 * 1024 * 1024
    max_image_pixels: int = 24_000_000
    max_frames: int = 8
    sensitivity_default: Sensitivity = Sensitivity.BALANCED
    embedding_enabled: bool = False
    embedding_model_path: str = ""
    #: Max per-guild hash indexes held resident (LRU); least-recently-used are
    #: evicted and rebuilt on demand. Sized for very large fleets.
    detection_guild_index_cap: int = Field(default=1024, ge=1)

    # Bus back-pressure / redelivery
    #: Max messages a single detection replica processes concurrently. Also set
    #: as the JetStream consumer's ``max_ack_pending`` so the server stops
    #: handing out more until acks catch up; under an image flood the surplus
    #: buffers in JetStream rather than ballooning replica memory. Default 10 is
    #: informed by the load harness: a 2 vCPU replica saturates near ~10 img/s,
    #: so deeper in-flight only adds latency and redelivery risk, not throughput.
    detection_max_inflight: int = Field(default=10, ge=1)
    #: Pull-fetch batch size for the detection consumer (clamped to the spare
    #: in-flight budget each loop).
    detection_fetch_batch: int = Field(default=16, ge=1)
    #: Seconds JetStream waits for an ack before redelivering. Must exceed the
    #: slowest expected handler (decode + hash of ``max_frames``) so slow
    #: processing buffers instead of triggering a spurious redelivery storm.
    detection_ack_wait_seconds: float = Field(default=60.0, gt=0.0)
    #: Max delivery attempts before JetStream gives up on a message.
    detection_max_deliver: int = Field(default=5, ge=1)
    #: JetStream publish-dedup window (seconds) applied to the events stream.
    bus_duplicate_window_seconds: float = Field(default=2 * 60 * 60, gt=0.0)
    #: Rows the outbox relay publishes per drain before yielding.
    detection_outbox_batch: int = Field(default=100, ge=1)
    #: Seconds the outbox relay sleeps when the outbox is idle.
    detection_outbox_poll_seconds: float = Field(default=0.5, gt=0.0)

    # Rate limiting
    #: Limiter backend. ``memory`` (default) is per-process and correct for a
    #: single replica; ``redis`` shares one bucket across replicas so effective
    #: limits do not multiply with replica count. Defaulting to ``memory`` means
    #: single-node self-hosters see zero behavioural change.
    ratelimit_backend: RateLimitBackend = RateLimitBackend.MEMORY
    ratelimit_redis_prefix: str = "optimus:rl"
    #: Opportunistic idle-bucket sweep cadence for the interactions in-memory
    #: rate-limiter fallback (seconds), used only when Redis is unavailable.
    interactions_inmemory_sweep_seconds: float = Field(default=300.0, gt=0.0)

    # Moderation
    #: Confidence at or above which a verdict is queued for moderator review.
    mod_queue_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Confidence at or above which the configured action is auto-applied.
    mod_auto_act_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    #: Per-guild Discord REST action budget (token bucket).
    mod_action_rate_capacity: float = 5.0
    mod_action_rate_refill: float = 1.0
    #: Cooldown between DM warnings to the same user (seconds).
    mod_dm_cooldown_seconds: int = Field(default=3600, ge=1)
    #: Default timeout applied by DELETE_TIMEOUT (seconds).
    mod_timeout_seconds: int = Field(default=3600, ge=1)
    mod_circuit_failure_threshold: int = Field(default=5, ge=1)
    mod_circuit_recovery_seconds: float = 30.0

    # Priority-aware action dispatch (single-replica scheduler in front of the
    # action executor). Protective work (delete scam / timeout / ban) is
    # dispatched ahead of courtesy work (appeal DMs, notifications) under load.
    #: Concurrent in-flight dispatch slots per replica. The Redis token bucket
    #: still bounds the actual REST call rate; this caps local concurrency.
    mod_dispatch_concurrency: int = Field(default=4, ge=1)
    #: Pending-queue cap per replica. COURTESY/NOTIFY submissions are rejected
    #: (reason-labeled metric) once full; PROTECT is always admitted.
    mod_dispatch_max_queue: int = Field(default=1000, ge=1)
    #: Wait time (seconds) that ages a queued item up one priority class — the
    #: starvation guard so low-priority work always drains under sustained load.
    mod_dispatch_aging_seconds: float = Field(default=5.0, gt=0.0)

    # Safe mode (anomaly-driven auto report-only)
    #: Multiplier of standard deviations above baseline that trips safe mode.
    safemode_sigma: float = Field(default=4.0, gt=0.0)
    #: EWMA smoothing factor for the rolling baseline (0..1).
    safemode_alpha: float = Field(default=0.3, gt=0.0, le=1.0)
    #: Minimum baseline mean before safe mode can trip (avoids small-sample noise).
    safemode_min_floor: float = Field(default=5.0, ge=0.0)
    #: Lifetime of the baseline state in Redis (seconds).
    safemode_ttl_seconds: int = Field(default=7 * 24 * 3600, ge=1)

    # Evidence storage
    evidence_enabled: bool = False
    evidence_ttl_seconds: int = Field(default=3600, ge=1, le=86_400)
    evidence_max_ttl_seconds: int = Field(default=86_400, ge=1, le=86_400)
    evidence_bucket: str = "optimus-evidence"
    evidence_endpoint_url: str = ""
    evidence_region: str = "us-east-1"
    evidence_sse: str = "AES256"
    evidence_presign_seconds: int = Field(default=300, ge=1, le=3600)

    # Data retention (deployment-wide cleanup job).
    #: Age in days past which detection/appeal rows are purged by the scheduler's
    #: ``retention_purge`` job. ``None`` (the default) disables the job entirely
    #: so self-hosters keep everything; set a positive value to enable bounded
    #: cleanup. This is the operator-level floor and is independent of the
    #: per-guild ``retention_days`` config consumed by the legacy
    #: :func:`enforce_retention` job.
    detection_retention_days: int | None = Field(default=None, ge=1)
    #: Rows deleted per DELETE statement, looping until a batch comes back
    #: short. Bounded batches keep locks/transactions short on huge tables.
    retention_batch_size: int = Field(default=1000, ge=1)
    #: Seconds to sleep between batches, yielding to foreground traffic and
    #: giving autovacuum room. Zero disables the pause (useful in tests).
    retention_batch_pause_seconds: float = Field(default=0.5, ge=0.0)

    # Scheduler intervals (seconds)
    scheduler_retention_interval: int = Field(default=3600, ge=1)
    scheduler_evidence_interval: int = Field(default=600, ge=1)
    scheduler_rollup_interval: int = Field(default=900, ge=1)
    scheduler_index_rebuild_interval: int = Field(default=1800, ge=1)
    scheduler_health_interval: int = Field(default=300, ge=1)
    #: Cadence of the deployment-wide batched retention purge (seconds).
    scheduler_retention_purge_interval: int = Field(default=86400, ge=1)
    scheduler_jitter_fraction: float = Field(default=0.1, ge=0.0, le=1.0)

    # Global hash DB signing (Ed25519, base64-encoded)
    #: Legacy single public key; also verifies pre-versioning (unsigned-id)
    #: records. Kept for backward compatibility.
    global_signing_public_key: str = ""
    global_signing_private_key: str = ""
    #: The active signing key's id. When set, new promotions are signed as
    #: ``"{key_id}:{sig}"``; blank keeps the legacy bare-signature format.
    global_signing_key_id: str = ""
    #: Currently-valid verification keys as ``"id:b64,id2:b64"``. Holding more
    #: than one enables key rotation with an overlap window.
    global_signing_public_keys: str = ""
    #: Comma-separated key ids that must never be trusted (revocation), even if
    #: still listed in ``global_signing_public_keys``.
    global_revoked_key_ids: str = ""

    # Global hash DB promotion (anti-sybil)
    #: Distinct (trusted) guilds required to promote a candidate fleet-wide.
    global_promotion_min_distinct: int = Field(default=MIN_DISTINCT_APPROVERS, ge=1)
    #: Optional allowlist of guild ids whose approvals corroborate a promotion,
    #: as a comma-separated list. Blank trusts every guild (relying on the
    #: distinct-count floor alone); setting it blocks sybil guilds entirely.
    global_trusted_guild_ids: str = ""

    @field_validator("shard_count", mode="before")
    @classmethod
    def _empty_shard_count_is_none(cls, value: object) -> object:
        """Treat a blank ``OPTIMUS_SHARD_COUNT`` env var as unset.

        ``.env.example`` ships the key present-but-empty; without this an empty
        string would fail int coercion instead of meaning "use hikari defaults".
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("detection_retention_days", mode="before")
    @classmethod
    def _empty_retention_days_is_none(cls, value: object) -> object:
        """Treat a blank ``OPTIMUS_DETECTION_RETENTION_DAYS`` env var as unset.

        ``.env.example`` ships the key present-but-empty; without this an empty
        string would fail int coercion at startup instead of meaning "retain
        everything" (the ``None`` default).
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("shard_ids", mode="before")
    @classmethod
    def _coerce_shard_ids(cls, value: object) -> object:
        """Parse the ``OPTIMUS_SHARD_IDS`` spec string into a tuple of ids.

        Non-string values (e.g. a tuple supplied directly in tests, or ``None``)
        pass through unchanged for the normal field machinery to validate.
        """
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            parsed = _parse_shard_ids(stripped)
            if not parsed:
                raise ValueError("shard_ids resolved to an empty set")
            return parsed
        return value

    @model_validator(mode="after")
    def _validate_inline_cap(self) -> Self:
        """Keep the inline-payload cap at or below the download cap.

        The inline base64 payload is what actually rides through NATS; allowing
        it to exceed the streamed download cap would defeat the bound.
        """
        if self.ingest_max_inline_bytes > self.ingest_max_bytes:
            raise ValueError(
                "ingest_max_inline_bytes must be <= ingest_max_bytes "
                f"({self.ingest_max_inline_bytes} > {self.ingest_max_bytes})"
            )
        return self

    @model_validator(mode="after")
    def _validate_sharding(self) -> Self:
        """Enforce that the per-replica shard subset is consistent with the fleet."""
        if self.shard_ids is None:
            return self
        if not self.shard_ids:
            raise ValueError("shard_ids must be non-empty when set")
        if self.shard_count is None:
            raise ValueError("shard_count is required when shard_ids is set")
        if any(i >= self.shard_count for i in self.shard_ids):
            raise ValueError(
                f"every shard id must be < shard_count ({self.shard_count}); "
                f"got {list(self.shard_ids)}"
            )
        return self

    @property
    def signing_public_key_map(self) -> dict[str, str]:
        """The currently-valid verification keys as an id -> base64 map."""
        return _parse_key_map(self.global_signing_public_keys)

    @property
    def revoked_signing_key_ids(self) -> frozenset[str]:
        """Key ids that must never be trusted, even if still listed."""
        return _parse_str_csv(self.global_revoked_key_ids)

    @property
    def trusted_guild_id_set(self) -> frozenset[int] | None:
        """The promotion trust allowlist, or ``None`` when unset (trust all)."""
        parsed = _parse_int_csv(self.global_trusted_guild_ids)
        return parsed or None

    @property
    def is_multi_tenant(self) -> bool:
        """Whether multi-tenant (SaaS) mode is active."""
        return self.tenancy is Tenancy.MULTI

    @property
    def is_simple_mode(self) -> bool:
        """Whether single-process ``simple`` mode is active."""
        return self.mode is Mode.SIMPLE

    @property
    def effective_database_url(self) -> str:
        """The DB URL for the active mode: SQLite in simple, configured otherwise."""
        return self.simple_database_url if self.is_simple_mode else self.database_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
