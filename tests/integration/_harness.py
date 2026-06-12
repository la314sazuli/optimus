"""Shared scaffolding for the in-process detection->moderation integration tests.

These helpers wire the *real* detection and moderation components together with
fake-but-faithful edges: an in-memory event bus that dispatches synchronously
(no NATS, no sleeps), a recording Discord REST double implementing the executor's
:class:`~optimus.services.moderation.actions.RestActions` surface, and a
fakeredis-backed Redis. The database is a real aiosqlite engine with the full
schema, exercised through the production repositories.
"""

from __future__ import annotations

import base64
import io
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from optimus.contracts.events import ImageFetchedEvent
from optimus.hashing import perceptual
from optimus.hashing.decoder import decode

Handler = Callable[[BaseModel], Awaitable[None]]


class InMemoryBus:
    """A synchronous in-process stand-in for :class:`~optimus.bus.nats.EventBus`.

    ``publish`` records every event and immediately awaits each handler
    subscribed to that subject, so a publish from one service drives the next
    service's consumer within the same call — deterministic, ordered, no polling
    loop and no background tasks. Handler exceptions surface to the publisher
    (the real bus would nak/redeliver); tests that need a swallowed failure wrap
    their own handler.
    """

    def __init__(self) -> None:
        self.published: list[tuple[str, BaseModel]] = []
        self._subs: dict[str, list[Handler]] = {}

    def subscribe(self, subject: str, handler: Handler) -> None:
        """Register ``handler`` to receive every event published to ``subject``."""
        self._subs.setdefault(subject, []).append(handler)

    async def publish(self, subject: str, event: BaseModel) -> None:
        """Record ``event`` and fan it out to every subscriber of ``subject``."""
        self.published.append((subject, event))
        for handler in list(self._subs.get(subject, ())):
            await handler(event)

    def events(self, subject: str) -> list[BaseModel]:
        """Return every event published to ``subject``, in order."""
        return [e for s, e in self.published if s == subject]


@dataclass
class RecordingRest:
    """A faithful :class:`RestActions` double that records every Discord call.

    Each method appends a ``(verb, args)`` tuple so a test can assert the exact
    enforcement steps (delete -> punitive -> DM) the executor performed. Any
    method name listed in ``fail_on`` raises to simulate a Discord outage,
    driving the executor's circuit breaker / backoff paths.
    """

    calls: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    fail_on: frozenset[str] = frozenset()
    dm_raises: bool = False

    def _record(self, verb: str, *args: int) -> None:
        self.calls.append((verb, args))
        if verb in self.fail_on:
            raise RuntimeError(f"discord_unavailable:{verb}")

    @property
    def verbs(self) -> list[str]:
        """The ordered list of REST verbs invoked (without arguments)."""
        return [verb for verb, _ in self.calls]

    async def delete_message(self, channel_id: int, message_id: int) -> None:
        self._record("delete_message", channel_id, message_id)

    async def timeout_member(self, guild_id: int, user_id: int, seconds: int) -> None:
        self._record("timeout_member", guild_id, user_id, seconds)

    async def kick_member(self, guild_id: int, user_id: int, reason: str) -> None:
        self._record("kick_member", guild_id, user_id)

    async def ban_member(self, guild_id: int, user_id: int, reason: str) -> None:
        self._record("ban_member", guild_id, user_id)

    async def unban_member(self, guild_id: int, user_id: int, reason: str) -> None:
        self._record("unban_member", guild_id, user_id)

    async def send_dm(self, user_id: int, content: str) -> None:
        self.calls.append(("send_dm", (user_id,)))
        if self.dm_raises:
            raise RuntimeError("closed_dms")


def single_session_scope(session: AsyncSession):  # type: ignore[no-untyped-def]
    """Wrap one live session as the async-context-manager factory the code expects.

    The integration flows share a single transaction so assertions can read rows
    written by an earlier step without committing between services (mirroring the
    pattern used by ``tests/integration/test_index_manager.py``).
    """

    @asynccontextmanager
    async def scope() -> AsyncIterator[AsyncSession]:
        yield session

    return scope


def make_scam_png(seed: int = 7, size: int = 64) -> bytes:
    """Render a deterministic noise PNG that stands in for a scam-campaign image.

    High-entropy noise gives the four perceptual hashes well-separated values, so
    an exact re-upload matches the registered hash at distance 0 while an
    *independent* clean image lands far outside the BK-tree candidate radius.
    """
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def hashes_for(data: bytes) -> dict[str, int]:
    """Decode ``data`` through the real sandboxed decoder and hash its first frame."""
    decoded = decode(data)
    assert decoded is not None, "fixture image failed to decode"
    return perceptual.compute_all(decoded.frames[0])


def image_fetched_event(
    data: bytes,
    *,
    guild_id: int,
    uploader_id: int,
    idempotency_key: str,
    message_id: int = 555,
    channel_id: int = 222,
    attachment_id: int = 444,
    correlation_id: str = "corr-int",
) -> ImageFetchedEvent:
    """Build an :class:`ImageFetchedEvent` carrying ``data`` inline as base64."""
    return ImageFetchedEvent(
        correlation_id=correlation_id,
        occurred_at=datetime.now(UTC),
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        attachment_id=attachment_id,
        uploader_id=uploader_id,
        idempotency_key=idempotency_key,
        content_type="image/png",
        size_bytes=len(data),
        sha256="0" * 64,
        data_b64=base64.b64encode(data).decode(),
    )
