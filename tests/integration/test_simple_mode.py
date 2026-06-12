"""Composition test for ``OPTIMUS_MODE=simple``.

This boots the *real* :class:`~optimus.app.simple.SimpleApp` — the in-process
bus, the in-memory store, a temp-file SQLite brought up by the real
``run_migrations`` (alembic upgrade head), and every service wired exactly as a
live simple-mode process would wire them. The only doubles are the Discord edges
that need a network: a recording REST (so we can assert the enforcement actions)
and a fake image fetcher (so ingest does not hit the network). No NATS, no Redis,
no Postgres.

A synthetic ``message_image.v1`` event — what the gateway publishes for one
attachment — is put on the bus, and we assert it flows the whole pipeline
(ingest fetches -> detection matches the registered scam hash -> verdict ->
moderation bans the uploader) and then that the app shuts down cleanly.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio

from optimus.app.simple import SimpleApp
from optimus.contracts.events import SUBJECT_MESSAGE_IMAGE, MessageImageEvent
from optimus.core.config import Settings
from optimus.core.ratelimit import InMemoryRateLimiter, RateLimit
from optimus.db.models import Guild, GuildHash
from optimus.db.repositories import (
    DetectionRepository,
    GuildHashRepository,
    GuildRepository,
)
from optimus.ingest.fetcher import FetchedImage
from optimus.services.ingest.worker import IngestWorker
from tests.integration._harness import RecordingRest, hashes_for, make_scam_png

BOT_USER_ID = 999
GUILD_OWNER_ID = 1


class _SimpleRest(RecordingRest):
    """A recording REST that also answers the reads the target resolver issues.

    The moderation coordinator's real ``_resolve_target`` fetches the member,
    guild, roles, and the bot's own member to compute the boundary context.
    Simple mode uses that real resolver, so the composition test must supply a
    REST double that returns a non-privileged, out-ranked uploader (so punitive
    actions are allowed) on top of recording the enforcement calls.
    """

    async def fetch_member(self, guild_id: int, user_id: int):  # type: ignore[no-untyped-def]
        # The bot carries the elevated role (so it outranks the target); the
        # uploader carries none (top role position 0).
        return _FakeMember(role_ids=(_BOT_ROLE_ID,) if user_id == BOT_USER_ID else ())

    async def fetch_guild(self, guild_id: int):  # type: ignore[no-untyped-def]
        return _FakeGuild(owner_id=GUILD_OWNER_ID)

    async def fetch_roles(self, guild_id: int):  # type: ignore[no-untyped-def]
        return (_FakeRole(role_id=_BOT_ROLE_ID, position=5, permissions=0),)


_BOT_ROLE_ID = 10


class _FakeMember:
    def __init__(self, role_ids: tuple[int, ...]) -> None:
        self.role_ids = role_ids


class _FakeGuild:
    def __init__(self, owner_id: int) -> None:
        self.owner_id = owner_id


class _FakeRole:
    def __init__(self, role_id: int, position: int, permissions: int) -> None:
        self.id = role_id
        self.position = position
        self.permissions = permissions


GUILD_ID = 4242
SCAM_UPLOADER = 9001
CAMPAIGN = "camp-simple"


def _settings(tmp_path) -> Settings:  # type: ignore[no-untyped-def]
    """Simple-mode settings pointing at a fresh temp-file SQLite."""
    db = tmp_path / "simple.db"
    return Settings(
        mode="simple",
        simple_database_url=f"sqlite+aiosqlite:///{db}",
        discord_token="test-token",
    )


def _fake_ingest_worker(settings: Settings, data: bytes) -> IngestWorker:
    """An ingest worker whose fetch returns ``data`` without touching the network."""

    async def fetch(_url: str) -> FetchedImage:
        return FetchedImage(data=data, content_type="image/png", final_url="https://x/y.png")

    return IngestWorker(
        fetch,
        InMemoryRateLimiter(),
        rate=RateLimit(capacity=100.0, refill_rate=100.0),
        max_inline_bytes=settings.ingest_max_inline_bytes,
    )


@pytest_asyncio.fixture
async def app(tmp_path) -> AsyncIterator[tuple[SimpleApp, RecordingRest]]:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    rest = _SimpleRest()
    application = await SimpleApp.build(settings, rest=rest, bot_user_id=BOT_USER_ID)
    await application.dispatcher.start()
    try:
        yield application, rest
    finally:
        await application.aclose()


async def _seed(app: SimpleApp, scam: bytes) -> None:
    """Create the guild policy row and register the scam campaign hash."""
    async with app._scope() as session:  # type: ignore[attr-defined]
        await GuildRepository(session).upsert(
            Guild(
                guild_id=GUILD_ID,
                action_policy="delete_ban",
                mod_queue_threshold=0.5,
                sensitivity="balanced",
                review_channel_id=None,
            )
        )
        h = hashes_for(scam)
        await GuildHashRepository(session, GUILD_ID).add(
            GuildHash(
                guild_id=GUILD_ID,
                hash_id=CAMPAIGN,
                phash=h["phash"],
                dhash=h["dhash"],
                whash=h["whash"],
                ahash=h["ahash"],
                source="local",
                status="active",
            )
        )


def _message_image_event() -> MessageImageEvent:
    return MessageImageEvent(
        correlation_id="corr-simple",
        occurred_at=datetime.now(UTC),
        guild_id=GUILD_ID,
        channel_id=222,
        message_id=555,
        attachment_id=444,
        uploader_id=SCAM_UPLOADER,
        url="https://cdn.example/scam.png",
        filename="scam.png",
        content_type="image/png",
    )


async def test_synthetic_image_flows_end_to_end_then_clean_shutdown(
    app: tuple[SimpleApp, RecordingRest],
) -> None:
    application, rest = app
    scam = make_scam_png()

    # Swap in a non-networked fetcher so ingest returns our scam bytes.
    application.ingest_worker = _fake_ingest_worker(application.settings, scam)
    await _seed(application, scam)

    # Signal completion off the *end* of the moderation handler rather than the
    # ban REST call: handle_verdict runs execute -> audit (the DB write the
    # assertions below read) -> report, so waiting on on_verdict returning is the
    # one point where both the ban and its persisted detection row are guaranteed
    # to exist. (Hooking ban_member would race the audit write.)
    pipeline_done = asyncio.Event()
    real_on_verdict = application.moderation.on_verdict

    async def on_verdict_then_signal(event: object) -> None:
        try:
            await real_on_verdict(event)  # type: ignore[arg-type]
        finally:
            pipeline_done.set()

    application.moderation.on_verdict = on_verdict_then_signal  # type: ignore[method-assign]

    application.start_pipeline()

    # One gateway-style event drives ingest -> detection -> moderation.
    await application.bus.publish(SUBJECT_MESSAGE_IMAGE, _message_image_event())

    # The pipeline runs across several async bus hops (ingest -> detection ->
    # moderation), each on its own task. Await the terminal completion signal
    # rather than polling a wall-clock deadline, so a slow CI host just waits
    # longer instead of failing. The generous outer timeout only guards against a
    # genuine hang (a step that never completes), not against being slow.
    try:
        await asyncio.wait_for(pipeline_done.wait(), timeout=30.0)
    except TimeoutError:
        pytest.fail(f"pipeline did not complete within 30s; saw calls={rest.calls}")

    # The enforcement order is delete-then-ban, against the right guild/uploader.
    assert rest.verbs[:2] == ["delete_message", "ban_member"]
    assert ("ban_member", (GUILD_ID, SCAM_UPLOADER)) in rest.calls

    # The detection was persisted to the shared SQLite with the action recorded.
    async with application._scope() as session:  # type: ignore[attr-defined]
        detections = await DetectionRepository(session, GUILD_ID).list_recent()
    assert any(d.action_taken == "delete_ban" for d in detections)


async def test_build_runs_migrations_and_is_ready(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A freshly built app has a working schema (migrations ran) and the readiness
    # DB check passes against the SQLite engine — proving the zero-dependency boot.
    settings = _settings(tmp_path)
    application = await SimpleApp.build(settings, bot_user_id=0)
    try:
        async with application._scope() as session:  # type: ignore[attr-defined]
            assert await GuildRepository(session).get(123) is None
    finally:
        await application.aclose()
