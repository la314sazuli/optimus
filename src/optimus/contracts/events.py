"""Versioned event contracts and NATS subject constants.

All inter-service messages are validated Pydantic models. Subjects are
versioned (``...v1``) so schemas can evolve without breaking consumers.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# --- NATS subjects (versioned) -------------------------------------------------

SUBJECT_MESSAGE_IMAGE = "events.message_image.v1"
SUBJECT_IMAGE_FETCHED = "events.image_fetched.v1"
SUBJECT_VERDICT = "events.verdict.v1"
SUBJECT_ACTION_RESULT = "events.action_result.v1"
SUBJECT_SWARM_ALERT = "events.swarm_alert.v1"
SUBJECT_GUILD_JOINED = "events.guild_joined.v1"
#: Terminal sink for messages that exhausted ``max_deliver`` on their original
#: subject. Nothing consumes it by default; an operator/alert drains it so a
#: poison or persistently-failing message is captured rather than silently lost.
SUBJECT_DEAD_LETTER = "events.dead_letter.v1"

#: Core-NATS (non-JetStream) pub/sub subject for hash-index invalidation.
SUBJECT_INDEX_INVALIDATE = "control.index_invalidate.v1"

#: Stream name carrying every ``events.*`` subject.
STREAM_EVENTS = "OPTIMUS_EVENTS"

#: All subjects bound to :data:`STREAM_EVENTS`.
EVENT_SUBJECTS: tuple[str, ...] = (
    SUBJECT_MESSAGE_IMAGE,
    SUBJECT_IMAGE_FETCHED,
    SUBJECT_VERDICT,
    SUBJECT_ACTION_RESULT,
    SUBJECT_SWARM_ALERT,
    SUBJECT_GUILD_JOINED,
    SUBJECT_DEAD_LETTER,
)


# --- Enumerations --------------------------------------------------------------


class Verdict(StrEnum):
    """Outcome of the detection pipeline for one image."""

    CLEAN = "clean"
    AMBIGUOUS = "ambiguous"
    SCAM = "scam"
    NON_DECISION = "non_decision"


class Action(StrEnum):
    """Moderation action applied to a detection."""

    NONE = "none"
    REPORT_ONLY = "report_only"
    DELETE = "delete"
    DELETE_TIMEOUT = "delete_timeout"
    DELETE_KICK = "delete_kick"
    DELETE_BAN = "delete_ban"


# --- Base ----------------------------------------------------------------------


class _Event(BaseModel):
    """Shared base for all events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = 1
    correlation_id: str
    occurred_at: datetime


# --- Event payloads ------------------------------------------------------------


class MessageImageEvent(_Event):
    """A message attachment that should be inspected. Subject: ``message_image.v1``."""

    guild_id: int
    channel_id: int
    message_id: int
    attachment_id: int
    uploader_id: int
    url: str
    filename: str
    content_type: str | None = None
    is_bot: bool = False
    is_webhook: bool = False


class ImageFetchedEvent(_Event):
    """A validated, in-bounds image ready for decoding. Subject: ``image_fetched.v1``.

    The raw image bytes ride inline as base64 (``data_b64``) so detection workers
    never touch disk. Streams are bounded, so the inline payload is acceptable for
    the small images this pipeline accepts.
    """

    guild_id: int
    channel_id: int
    message_id: int
    attachment_id: int
    uploader_id: int
    idempotency_key: str
    content_type: str
    size_bytes: int
    sha256: str
    data_b64: str


class HashSet(BaseModel):
    """The perceptual hash ensemble for one image (each a 64-bit value)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phash: int = Field(ge=0, lt=1 << 64)
    dhash: int = Field(ge=0, lt=1 << 64)
    whash: int = Field(ge=0, lt=1 << 64)
    ahash: int = Field(ge=0, lt=1 << 64)


class VerdictEvent(_Event):
    """The detection outcome for one image. Subject: ``verdict.v1``."""

    guild_id: int
    channel_id: int
    message_id: int
    attachment_id: int
    uploader_id: int
    idempotency_key: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    hashes: HashSet | None = None
    matched_hash_id: str | None = None
    distances: dict[str, int] = Field(default_factory=dict)


class ActionResultEvent(_Event):
    """The result of applying a moderation action. Subject: ``action_result.v1``."""

    guild_id: int
    channel_id: int
    message_id: int
    attachment_id: int
    uploader_id: int
    idempotency_key: str
    action: Action
    success: bool
    detail: str | None = None


class IndexInvalidateEvent(_Event):
    """Signals detection workers to reload a guild's (or the global) hash index.

    Subject: ``control.index_invalidate.v1``. A ``guild_id`` of ``None`` means the
    global promoted-hash set changed and every worker should reload it.
    """

    guild_id: int | None = None


class SwarmAlertEvent(_Event):
    """A cross-guild swarm correlation alert. Subject: ``swarm_alert.v1``."""

    phash: int = Field(ge=0, lt=1 << 64)
    distinct_guilds: int = Field(ge=1)
    window_seconds: int = Field(ge=1)
    sample_guild_ids: list[int] = Field(default_factory=list)


class GuildJoinedEvent(_Event):
    """The bot was added to a guild. Subject: ``guild_joined.v1``.

    Triggers one-time provisioning such as the private mod-review channel.
    """

    guild_id: int
    guild_name: str | None = None
    owner_id: int | None = None


class DeadLetterEvent(_Event):
    """A message that exhausted ``max_deliver`` on its origin. Subject: ``dead_letter.v1``.

    Carries the original subject, the raw (still-JSON) payload, how many times it
    was delivered, and the last handler error, so the failure is captured for
    inspection/replay instead of being dropped when JetStream gives up.
    """

    original_subject: str
    payload: str
    delivered: int = Field(ge=1)
    error: str | None = None
