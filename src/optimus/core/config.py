"""Application configuration loaded from the environment (prefix ``OPTIMUS_``)."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    tenancy: Tenancy = Tenancy.SINGLE

    # Discord
    discord_token: str = ""
    discord_client_id: str = ""
    discord_client_secret: str = ""

    # Datastores
    database_url: str = "postgresql+asyncpg://optimus:optimus@localhost:5432/optimus"
    redis_url: str = "redis://localhost:6379/0"
    nats_url: str = "nats://localhost:4222"

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
    ingest_max_redirects: int = 3
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

    # Scheduler intervals (seconds)
    scheduler_retention_interval: int = Field(default=3600, ge=1)
    scheduler_evidence_interval: int = Field(default=600, ge=1)
    scheduler_rollup_interval: int = Field(default=900, ge=1)
    scheduler_index_rebuild_interval: int = Field(default=1800, ge=1)
    scheduler_health_interval: int = Field(default=300, ge=1)
    scheduler_jitter_fraction: float = Field(default=0.1, ge=0.0, le=1.0)

    # Global hash DB signing (Ed25519, base64-encoded)
    global_signing_public_key: str = ""
    global_signing_private_key: str = ""

    @property
    def is_multi_tenant(self) -> bool:
        """Whether multi-tenant (SaaS) mode is active."""
        return self.tenancy is Tenancy.MULTI


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
