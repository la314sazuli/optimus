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

    # Rate limiting
    ratelimit_redis_prefix: str = "optimus:rl"

    # Evidence storage
    evidence_enabled: bool = False
    evidence_ttl_seconds: int = Field(default=3600, ge=1, le=86_400)

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
