"""Pure helpers for turning a Discord message into image events.

Kept free of hikari types so the extraction and filtering rules can be unit
tested directly. The bot module adapts hikari objects into these plain inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from optimus.contracts.events import MessageImageEvent

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
_IMAGE_CONTENT_PREFIX = "image/"
# A conservative URL matcher for image links embedded in message content.
_URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Attachment:
    """A plain view of a Discord attachment."""

    id: int
    url: str
    filename: str
    content_type: str | None = None


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """A plain view of a message relevant to image scanning."""

    guild_id: int
    channel_id: int
    message_id: int
    author_id: int
    content: str = ""
    attachments: tuple[Attachment, ...] = ()
    embed_image_urls: tuple[str, ...] = ()
    is_bot: bool = False
    is_webhook: bool = False
    author_role_ids: frozenset[int] = field(default_factory=frozenset)


def _looks_like_image_url(url: str) -> bool:
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    return path.endswith(_IMAGE_EXT)


def _is_image_attachment(att: Attachment) -> bool:
    if att.content_type and att.content_type.lower().startswith(_IMAGE_CONTENT_PREFIX):
        return True
    return _looks_like_image_url(att.url)


def extract_image_urls_from_content(content: str) -> list[str]:
    """Return image-looking URLs found in free text message content."""
    return [m.group(0) for m in _URL_RE.finditer(content) if _looks_like_image_url(m.group(0))]


def build_events(
    msg: IncomingMessage, *, correlation_id: str, now: datetime | None = None
) -> list[MessageImageEvent]:
    """Build one :class:`MessageImageEvent` per inspectable image in ``msg``.

    Deduplicates by URL and synthesizes stable attachment ids for embedded
    content-URLs (which lack a Discord attachment id) from a hash of the URL.
    """
    occurred = now or datetime.now(UTC)
    events: list[MessageImageEvent] = []
    seen_urls: set[str] = set()

    for att in msg.attachments:
        if not _is_image_attachment(att) or att.url in seen_urls:
            continue
        seen_urls.add(att.url)
        events.append(
            MessageImageEvent(
                correlation_id=correlation_id,
                occurred_at=occurred,
                guild_id=msg.guild_id,
                channel_id=msg.channel_id,
                message_id=msg.message_id,
                attachment_id=att.id,
                uploader_id=msg.author_id,
                url=att.url,
                filename=att.filename,
                content_type=att.content_type,
                is_bot=msg.is_bot,
                is_webhook=msg.is_webhook,
            )
        )

    urls = list(msg.embed_image_urls) + extract_image_urls_from_content(msg.content)
    for url in urls:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        synthetic_id = _synthetic_attachment_id(msg.message_id, url)
        events.append(
            MessageImageEvent(
                correlation_id=correlation_id,
                occurred_at=occurred,
                guild_id=msg.guild_id,
                channel_id=msg.channel_id,
                message_id=msg.message_id,
                attachment_id=synthetic_id,
                uploader_id=msg.author_id,
                url=url,
                filename=url.rsplit("/", 1)[-1][:128] or "embed",
                content_type=None,
                is_bot=msg.is_bot,
                is_webhook=msg.is_webhook,
            )
        )
    return events


def _synthetic_attachment_id(message_id: int, url: str) -> int:
    """Derive a stable positive 63-bit id for an embedded (non-attachment) URL."""
    import hashlib

    digest = hashlib.sha256(f"{message_id}:{url}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
