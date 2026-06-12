"""SQLAlchemy 2.0 ORM models for the optimus data model.

Every per-guild table is keyed and indexed on ``guild_id``. In multi-tenant
mode Postgres RLS (see migration 0001) further isolates rows; repository classes
additionally enforce ``guild_id`` filtering in every query as defense in depth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Dialect,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

_UINT64_SPAN = 1 << 64
_INT64_MAX = (1 << 63) - 1


class Uint64(TypeDecorator[int]):
    """Store an unsigned 64-bit hash in a signed ``int8`` column.

    Perceptual hashes span the full ``[0, 2**64)`` range, but the underlying
    storage (Postgres ``bigint`` / SQLite ``INTEGER``) is *signed* int8, capped
    at ``2**63 - 1``. Any hash with the high bit set therefore overflows on
    insert. This decorator maps the value through its two's-complement
    representation on the way in and back on the way out, so the column stays a
    plain ``BIGINT`` (no schema change, indexes/joins unaffected) while callers
    only ever see the unsigned value the rest of the pipeline expects.
    """

    impl = BigInteger
    cache_ok = True

    def process_bind_param(self, value: int | None, dialect: Dialect) -> int | None:
        if value is None:
            return None
        return value - _UINT64_SPAN if value > _INT64_MAX else value

    def process_result_value(self, value: int | None, dialect: Dialect) -> int | None:
        if value is None:
            return None
        return value + _UINT64_SPAN if value < 0 else value


class Base(DeclarativeBase):
    """Declarative base for all models."""

    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON}


def _ts() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Guild(Base):
    """Per-guild configuration."""

    __tablename__ = "guilds"

    guild_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    sensitivity: Mapped[str] = mapped_column(String(16), default="balanced", nullable=False)
    action_policy: Mapped[str] = mapped_column(String(32), default="report_only", nullable=False)
    mod_queue_threshold: Mapped[float] = mapped_column(default=0.5, nullable=False)
    review_channel_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    retention_days: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    optin_global_db: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    optin_scan_bots: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    optin_evidence_storage: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    locale: Mapped[str] = mapped_column(String(8), default="en", nullable=False)
    safe_mode: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = _ts()


class GuildChannelIgnored(Base):
    """Channels excluded from scanning."""

    __tablename__ = "guild_channels_ignored"

    guild_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("guilds.guild_id", ondelete="CASCADE"), primary_key=True, index=True
    )
    channel_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)


class GuildRoleIgnored(Base):
    """Roles excluded from scanning."""

    __tablename__ = "guild_roles_ignored"

    guild_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("guilds.guild_id", ondelete="CASCADE"), primary_key=True, index=True
    )
    role_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)


class GuildTrustedUser(Base):
    """Users exempt from scanning."""

    __tablename__ = "guild_trusted_users"

    guild_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("guilds.guild_id", ondelete="CASCADE"), primary_key=True, index=True
    )
    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)


class GuildHash(Base):
    """A per-guild known scam-image hash."""

    __tablename__ = "guild_hashes"

    guild_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("guilds.guild_id", ondelete="CASCADE"), nullable=False, index=True
    )
    hash_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    phash: Mapped[int] = mapped_column(Uint64, nullable=False)
    dhash: Mapped[int] = mapped_column(Uint64, nullable=False)
    whash: Mapped[int] = mapped_column(Uint64, nullable=False)
    ahash: Mapped[int] = mapped_column(Uint64, nullable=False, default=0)
    # Hashes of the horizontally-flipped source image, populated when the image
    # was available at indexing time so mirrored re-shares match (NULL otherwise).
    mphash: Mapped[int | None] = mapped_column(Uint64, nullable=True)
    mdhash: Mapped[int | None] = mapped_column(Uint64, nullable=True)
    mwhash: Mapped[int | None] = mapped_column(Uint64, nullable=True)
    mahash: Mapped[int | None] = mapped_column(Uint64, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="local", nullable=False)
    added_by: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    created_at: Mapped[datetime] = _ts()


class GuildWhitelist(Base):
    """A per-guild whitelisted hash that always overrides global matches."""

    __tablename__ = "guild_whitelist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    guild_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("guilds.guild_id", ondelete="CASCADE"), nullable=False, index=True
    )
    phash: Mapped[int] = mapped_column(Uint64, nullable=False)
    dhash: Mapped[int] = mapped_column(Uint64, nullable=False)
    whash: Mapped[int] = mapped_column(Uint64, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    added_by: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = _ts()


class GlobalHash(Base):
    """A globally-shared scam-image hash (candidate / promoted / revoked)."""

    __tablename__ = "global_hashes"

    hash_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    phash: Mapped[int] = mapped_column(Uint64, nullable=False)
    dhash: Mapped[int] = mapped_column(Uint64, nullable=False)
    whash: Mapped[int] = mapped_column(Uint64, nullable=False)
    # Hashes of the horizontally-flipped source image (NULL when unavailable).
    mphash: Mapped[int | None] = mapped_column(Uint64, nullable=True)
    mdhash: Mapped[int | None] = mapped_column(Uint64, nullable=True)
    mwhash: Mapped[int | None] = mapped_column(Uint64, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="candidate", nullable=False, index=True)
    campaign_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitter_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    submitter_guild_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = _ts()


class GlobalSubmitter(Base):
    """A user's reputation as a contributor to the global hash database."""

    __tablename__ = "global_submitters"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    reputation: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    submitted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confirmed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rejected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = _ts()


class GlobalHashApproval(Base):
    """A moderator approval toward promoting a candidate global hash."""

    __tablename__ = "global_hash_approvals"
    __table_args__ = (UniqueConstraint("hash_id", "approver_user_id", name="uq_global_approval"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hash_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("global_hashes.hash_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    approver_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    approver_guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = _ts()


class Detection(Base):
    """A recorded detection event."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    channel_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    attachment_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    uploader_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    distances: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)
    action_taken: Mapped[str] = mapped_column(String(32), default="none", nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    created_at: Mapped[datetime] = _ts()


class Appeal(Base):
    """A user appeal against a detection."""

    __tablename__ = "appeals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    detection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("detections.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="open", nullable=False)
    resolved_by: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = _ts()


class ModAction(Base):
    """Audit log of every administrative action."""

    __tablename__ = "mod_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    actor_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = _ts()


class UserOptout(Base):
    """A user who has opted out of processing (right to erasure)."""

    __tablename__ = "users_optout"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = _ts()


class StatsRollup(Base):
    """Periodic per-guild statistics rollup."""

    __tablename__ = "stats_rollups"
    __table_args__ = (UniqueConstraint("guild_id", "bucket_start", name="uq_stats_bucket"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    bucket_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    detections: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    false_positives: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    actions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class Evidence(Base):
    """A reference to a stored (encrypted, TTL'd) evidence object."""

    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("detections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    object_key: Mapped[str] = mapped_column(String(256), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
