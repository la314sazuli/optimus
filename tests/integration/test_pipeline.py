"""End-to-end integration tests for the detection -> moderation pipeline.

These wire the *real* components together in-process: the perceptual hashing
ensemble, the BK-tree index, the pure policy/boundaries logic, the
:class:`DetectionWorker`, the :class:`ModerationCoordinator` + executor (with its
real circuit breaker, token-bucket rate limiter, and idempotency guard), and the
appeal/review handlers — over a fakeredis backend, an aiosqlite DB with the
production schema, and a synchronous in-memory bus. A single image event
published onto the bus drives detection, which publishes a verdict, which drives
moderation, which applies an action and records the audit trail; tests assert the
observable end-to-end behaviour at each stage.

Everything is deterministic: no real NATS, no sleeps, and the components that
depend on real time (the circuit breaker and the token bucket) run on injected
frozen clocks. The rate limiter is the in-process :class:`InMemoryRateLimiter`
rather than the Redis-Lua one, because fakeredis does not implement ``EVAL``; the
token-bucket semantics under test are identical.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession as _Session
from sqlalchemy.ext.asyncio import create_async_engine

from optimus.contracts.events import (
    SUBJECT_ACTION_RESULT,
    SUBJECT_IMAGE_FETCHED,
    SUBJECT_VERDICT,
    Action,
    ActionResultEvent,
    Verdict,
    VerdictEvent,
)
from optimus.core.circuit import CircuitBreaker
from optimus.core.config import Sensitivity, get_settings
from optimus.core.ratelimit import InMemoryRateLimiter, RateLimit
from optimus.db.engine import create_session_factory, session_scope
from optimus.db.models import Appeal, Base, Detection, Guild, GuildHash
from optimus.db.repositories import (
    DetectionRepository,
    GuildHashRepository,
    GuildRepository,
    ModActionRepository,
)
from optimus.services.detection.index import IndexManager
from optimus.services.detection.matcher import WhitelistEntry
from optimus.services.detection.service import DetectionService
from optimus.services.detection.worker import DetectionWorker
from optimus.services.interactions.handlers import (
    InteractionContext,
    handle_component,
)
from optimus.services.interactions.logic import ComponentAction, Permission
from optimus.services.interactions.service import DbDeps
from optimus.services.moderation.actions import ActionExecutor
from optimus.services.moderation.boundaries import TargetContext
from optimus.services.moderation.cooldown import Cooldown
from optimus.services.moderation.coordinator import GuildModConfig, ModerationCoordinator
from optimus.services.moderation.review import ReportData
from optimus.services.moderation.service import ModerationService
from tests.integration._harness import (
    InMemoryBus,
    RecordingRest,
    hashes_for,
    image_fetched_event,
    make_scam_png,
    single_session_scope,
)

GUILD_ID = 4242
SCAM_UPLOADER = 9001
MOD_USER = 7
REVIEW_CHANNEL = 88
CAMPAIGN = "camp-int"


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[_Session]:
    """A live aiosqlite session with the full production schema created."""
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    async with session_scope(factory) as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    """A fakeredis client (token bucket + idempotency + cooldown keyspace)."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


# --- wiring helpers ---------------------------------------------------------


async def _seed_guild(
    session: _Session,
    *,
    action_policy: str = "delete_ban",
    safe_mode: bool = False,
    review_channel_id: int | None = REVIEW_CHANNEL,
) -> None:
    """Create the guild config row that drives moderation policy."""
    await GuildRepository(session).upsert(
        Guild(
            guild_id=GUILD_ID,
            action_policy=action_policy,
            mod_queue_threshold=0.5,
            sensitivity="balanced",
            safe_mode=safe_mode,
            review_channel_id=review_channel_id,
        )
    )


async def _register_scam_hash(session: _Session, data: bytes) -> dict[str, int]:
    """Register ``data``'s hash set as a known guild scam hash; return the hashes."""
    h = hashes_for(data)
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
    return h


def _build_detection_service(
    bus: InMemoryBus, session: _Session, redis: object
) -> DetectionService:
    """Wire a real :class:`DetectionService` over the shared session + bus."""
    scope = single_session_scope(session)
    index_manager = IndexManager(scope)
    settings = get_settings()

    async def guild_index(guild_id: int):  # type: ignore[no-untyped-def]
        return await index_manager.guild_index(guild_id)

    async def global_index():  # type: ignore[no-untyped-def]
        return await index_manager.global_index()

    async def whitelist(guild_id: int) -> list[WhitelistEntry]:
        from optimus.db.repositories import WhitelistRepository

        rows = await WhitelistRepository(session, guild_id).list()
        return [WhitelistEntry(phash=r.phash) for r in rows]

    async def sensitivity(guild_id: int) -> Sensitivity:
        guild = await GuildRepository(session).get(guild_id)
        return Sensitivity(guild.sensitivity) if guild is not None else Sensitivity.BALANCED

    from optimus.core.idempotency import IdempotencyGuard

    worker = DetectionWorker(
        guild_index=guild_index,
        global_index=global_index,
        whitelist=whitelist,
        sensitivity=sensitivity,
        idempotency_acquire=IdempotencyGuard(redis).acquire,
        swarm=None,
    )
    return DetectionService(settings, bus, worker, index_manager, scope)  # type: ignore[arg-type]


def _build_moderation_service(
    bus: InMemoryBus,
    session: _Session,
    redis: object,
    *,
    rest: RecordingRest,
    breaker: CircuitBreaker | None = None,
    rate: RateLimit | None = None,
    target_owner_id: int = 1,
) -> ModerationService:
    """Wire a real :class:`ModerationService` with a recording REST + DB audit.

    Mirrors ``build_coordinator`` but injects the recording REST double, an
    optional pre-tripped breaker / tight rate limit, and a target resolver that
    returns a non-privileged member (so boundary checks pass and punitive actions
    are allowed to run).
    """
    scope = single_session_scope(session)
    settings = get_settings()

    # A process-local token bucket on a frozen clock: real token-bucket semantics
    # without the Redis Lua EVAL that fakeredis does not implement. The frozen
    # clock means tokens never refill mid-test, so a tight capacity starves
    # deterministically.
    limiter = InMemoryRateLimiter(time_source=_FakeClock().now)
    executor = ActionExecutor(
        rest,
        limiter,
        bot_user_id=999,
        rate=rate or RateLimit(capacity=5.0, refill_rate=1.0),
        idempotency_acquire=_ActionGuard(redis).acquire,
        dm_cooldown=Cooldown(redis, window_seconds=3600),
        breaker=breaker,
    )

    async def config(guild_id: int) -> GuildModConfig:
        guild = await GuildRepository(session).get(guild_id)
        action = Action(guild.action_policy) if guild is not None else Action.REPORT_ONLY
        return GuildModConfig(
            guild_id=guild_id,
            configured_action=action,
            mod_queue_threshold=guild.mod_queue_threshold if guild else 0.5,
            auto_act_threshold=settings.mod_auto_act_threshold,
            safe_mode=guild.safe_mode if guild else False,
            locale=guild.locale if guild else "en",
            review_channel_id=guild.review_channel_id if guild else None,
            timeout_seconds=settings.mod_timeout_seconds,
        )

    async def target(guild_id: int, user_id: int) -> TargetContext | None:
        # An ordinary, non-privileged member the bot outranks: boundary check passes.
        return TargetContext(
            user_id=user_id,
            guild_owner_id=target_owner_id,
            bot_user_id=999,
            is_administrator=False,
            top_role_position=1,
            bot_top_role_position=5,
        )

    review_posts: list[ReportData] = []

    async def report(channel_id: int, data: ReportData) -> int | None:
        review_posts.append(data)
        return 1000 + len(review_posts)

    async def audit(event: VerdictEvent, action: str, result) -> int | None:  # type: ignore[no-untyped-def]
        det_repo = DetectionRepository(session, event.guild_id)
        detection = await det_repo.get_by_idempotency_key(event.idempotency_key)
        if detection is None:
            detection = await det_repo.record(
                Detection(
                    guild_id=event.guild_id,
                    message_id=event.message_id,
                    channel_id=event.channel_id,
                    attachment_id=event.attachment_id,
                    uploader_id=event.uploader_id,
                    distances=dict(event.distances),
                    verdict=event.verdict.value,
                    idempotency_key=event.idempotency_key,
                )
            )
        await det_repo.set_action_taken(detection.id, action)
        await ModActionRepository(session, event.guild_id).record(
            actor_id=0,
            action=action,
            target=str(event.uploader_id),
            payload={"success": result.success, "detail": result.detail},
        )
        return detection.id

    coordinator = ModerationCoordinator(
        config=config, target=target, executor=executor, report=report, audit=audit
    )
    svc = ModerationService(settings, bus, coordinator, scope)  # type: ignore[arg-type]
    svc.review_posts = review_posts  # type: ignore[attr-defined]  # test inspection hook
    return svc


class _ActionGuard:
    """SET NX idempotency guard mirroring the production action guard."""

    def __init__(self, redis: object) -> None:
        self._redis = redis

    async def acquire(self, key: str) -> bool:
        result = await self._redis.set(key, "1", nx=True, ex=86_400)  # type: ignore[attr-defined]
        return result is True or result == "OK"


async def _wire_pipeline(
    session: _Session,
    redis: object,
    *,
    rest: RecordingRest,
    breaker: CircuitBreaker | None = None,
    rate: RateLimit | None = None,
) -> InMemoryBus:
    """Connect detection -> bus -> moderation so one image event drives both."""
    bus = InMemoryBus()
    detection = _build_detection_service(bus, session, redis)
    moderation = _build_moderation_service(
        bus, session, redis, rest=rest, breaker=breaker, rate=rate
    )
    bus.subscribe(SUBJECT_IMAGE_FETCHED, detection.on_image)
    bus.subscribe(SUBJECT_VERDICT, moderation.on_verdict)
    bus._moderation = moderation  # type: ignore[attr-defined]  # keep a handle for assertions
    return bus


# --- happy path -------------------------------------------------------------


async def test_known_scam_image_flows_to_ban_action_and_review(
    db_session: _Session, redis: fakeredis.aioredis.FakeRedis
) -> None:
    scam = make_scam_png()
    await _seed_guild(db_session, action_policy="delete_ban")
    await _register_scam_hash(db_session, scam)

    rest = RecordingRest()
    bus = await _wire_pipeline(db_session, redis, rest=rest)

    # One image event drives the whole pipeline synchronously.
    await bus.publish(
        SUBJECT_IMAGE_FETCHED,
        image_fetched_event(
            scam, guild_id=GUILD_ID, uploader_id=SCAM_UPLOADER, idempotency_key="scam-1"
        ),
    )

    # Detection emitted a SCAM verdict that matched the registered campaign.
    verdicts = bus.events(SUBJECT_VERDICT)
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert isinstance(verdict, VerdictEvent)
    assert verdict.verdict is Verdict.SCAM
    assert verdict.matched_hash_id == CAMPAIGN
    assert verdict.confidence >= get_settings().mod_auto_act_threshold

    # Moderation auto-acted: message deleted then uploader banned (then DM warned).
    assert rest.verbs[:2] == ["delete_message", "ban_member"]
    assert ("ban_member", (GUILD_ID, SCAM_UPLOADER)) in rest.calls

    # An action_result was emitted reflecting the successful ban.
    results = bus.events(SUBJECT_ACTION_RESULT)
    assert len(results) == 1
    assert isinstance(results[0], ActionResultEvent)
    assert results[0].action is Action.DELETE_BAN
    assert results[0].success is True

    # The detection + audit row were persisted, and a review report was produced.
    detection = await DetectionRepository(db_session, GUILD_ID).get_by_idempotency_key("scam-1")
    assert detection is not None
    assert detection.verdict == "scam"
    assert detection.action_taken == "delete_ban"
    audits = await ModActionRepository(db_session, GUILD_ID).list_recent()
    assert any(a.action == "delete_ban" and a.target == str(SCAM_UPLOADER) for a in audits)
    assert len(bus._moderation.review_posts) == 1  # type: ignore[attr-defined]
    assert bus._moderation.review_posts[0].detection_id == detection.id  # type: ignore[attr-defined]


# --- negative path ----------------------------------------------------------


async def test_clean_image_produces_no_action(
    db_session: _Session, redis: fakeredis.aioredis.FakeRedis
) -> None:
    # Register one scam campaign, then upload an unrelated (clean) image.
    await _seed_guild(db_session, action_policy="delete_ban")
    await _register_scam_hash(db_session, make_scam_png(seed=7))
    clean = make_scam_png(seed=999)  # independent noise -> far from the registered hash

    rest = RecordingRest()
    bus = await _wire_pipeline(db_session, redis, rest=rest)

    await bus.publish(
        SUBJECT_IMAGE_FETCHED,
        image_fetched_event(
            clean, guild_id=GUILD_ID, uploader_id=SCAM_UPLOADER, idempotency_key="clean-1"
        ),
    )

    verdict = bus.events(SUBJECT_VERDICT)[0]
    assert isinstance(verdict, VerdictEvent)
    assert verdict.verdict is Verdict.CLEAN
    assert verdict.matched_hash_id is None

    # No Discord enforcement happened, and the action result is the NONE no-op.
    assert rest.calls == []
    result = bus.events(SUBJECT_ACTION_RESULT)[0]
    assert isinstance(result, ActionResultEvent)
    assert result.action is Action.NONE
    # A clean verdict is not surfaced for review (Decision.NONE short-circuits).
    assert bus._moderation.review_posts == []  # type: ignore[attr-defined]


# --- safe mode --------------------------------------------------------------


async def test_safe_mode_suppresses_auto_action(
    db_session: _Session, redis: fakeredis.aioredis.FakeRedis
) -> None:
    # Same high-confidence scam as the happy path, but the guild is in safe mode.
    scam = make_scam_png()
    await _seed_guild(db_session, action_policy="delete_ban", safe_mode=True)
    await _register_scam_hash(db_session, scam)

    rest = RecordingRest()
    bus = await _wire_pipeline(db_session, redis, rest=rest)

    await bus.publish(
        SUBJECT_IMAGE_FETCHED,
        image_fetched_event(
            scam, guild_id=GUILD_ID, uploader_id=SCAM_UPLOADER, idempotency_key="safe-1"
        ),
    )

    # Detection still calls it a scam...
    assert bus.events(SUBJECT_VERDICT)[0].verdict is Verdict.SCAM  # type: ignore[union-attr]
    # ...but safe mode downgrades enforcement to report-only: no punitive REST call.
    assert "ban_member" not in rest.verbs
    assert "timeout_member" not in rest.verbs
    assert "kick_member" not in rest.verbs

    result = bus.events(SUBJECT_ACTION_RESULT)[0]
    assert isinstance(result, ActionResultEvent)
    assert result.action is Action.REPORT_ONLY
    # The detection is recorded as report_only (queued), and still surfaced for review.
    detection = await DetectionRepository(db_session, GUILD_ID).get_by_idempotency_key("safe-1")
    assert detection is not None
    assert detection.action_taken == "report_only"
    assert len(bus._moderation.review_posts) == 1  # type: ignore[attr-defined]


# --- appeal flow (Cycle 4 security fix) -------------------------------------


async def test_appeal_flow_owner_succeeds_nonowner_rejected_then_mod_reverses(
    db_session: _Session, redis: fakeredis.aioredis.FakeRedis
) -> None:
    # Run the happy path first so there is a real banned detection to appeal.
    scam = make_scam_png()
    await _seed_guild(db_session, action_policy="delete_ban")
    await _register_scam_hash(db_session, scam)
    rest = RecordingRest()
    bus = await _wire_pipeline(db_session, redis, rest=rest)
    await bus.publish(
        SUBJECT_IMAGE_FETCHED,
        image_fetched_event(
            scam, guild_id=GUILD_ID, uploader_id=SCAM_UPLOADER, idempotency_key="appeal-1"
        ),
    )
    detection = await DetectionRepository(db_session, GUILD_ID).get_by_idempotency_key("appeal-1")
    assert detection is not None and detection.action_taken == "delete_ban"

    # A process-local limiter avoids the Redis Lua EVAL the appeal-cooldown check
    # would otherwise issue (fakeredis has no EVAL); the cooldown semantics are
    # identical for a single click.
    deps = DbDeps(db_session, InMemoryRateLimiter(), get_settings())

    # A NON-owner pressing the appeal button is silently rejected (no appeal row).
    intruder_ctx = InteractionContext(
        guild_id=GUILD_ID, user_id=SCAM_UPLOADER + 1, member_permissions=0, command=""
    )
    rejected = await handle_component(intruder_ctx, ComponentAction.APPEAL_OPEN, detection.id, deps)
    assert rejected.i18n_key == "command.appeal_none"
    assert (await db_session.execute(Appeal.__table__.select())).fetchall() == []

    # The OWNER (the uploader the detection was filed against) can open an appeal.
    owner_ctx = InteractionContext(
        guild_id=GUILD_ID, user_id=SCAM_UPLOADER, member_permissions=0, command=""
    )
    opened = await handle_component(owner_ctx, ComponentAction.APPEAL_OPEN, detection.id, deps)
    assert opened.i18n_key == "dm.appeal_submitted"
    appeals = (await db_session.execute(Appeal.__table__.select())).fetchall()
    assert len(appeals) == 1
    appeal_id = appeals[0].id
    assert appeals[0].status == "open"
    assert appeals[0].user_id == SCAM_UPLOADER

    # A moderator (MANAGE_GUILD) approving the appeal reverses the action.
    mod_ctx = InteractionContext(
        guild_id=GUILD_ID,
        user_id=MOD_USER,
        member_permissions=int(Permission.MANAGE_GUILD),
        command="",
    )
    approved = await handle_component(mod_ctx, ComponentAction.APPEAL_APPROVE, appeal_id, deps)
    assert approved.i18n_key == "button.appeal_approved"

    appeal_row = await deps.get_appeal(GUILD_ID, appeal_id)
    assert appeal_row is not None and appeal_row["status"] == "approved"
    reversed_detection = await db_session.get(Detection, detection.id)
    assert reversed_detection is not None
    assert reversed_detection.action_taken == "reversed"


async def test_appeal_open_requires_owning_member_without_permissions(
    db_session: _Session, redis: fakeredis.aioredis.FakeRedis
) -> None:
    # Regression guard for the Cycle 4 fix: ownership, not permissions, gates an
    # appeal-open. A user with elevated perms but who did NOT upload the image is
    # still rejected, because the detection does not belong to them.
    await _seed_guild(db_session, action_policy="delete_ban")
    det = await DetectionRepository(db_session, GUILD_ID).record(
        Detection(
            guild_id=GUILD_ID,
            message_id=1,
            channel_id=2,
            attachment_id=3,
            uploader_id=SCAM_UPLOADER,
            distances={},
            verdict="scam",
            idempotency_key="own-1",
        )
    )
    deps = DbDeps(db_session, InMemoryRateLimiter(), get_settings())
    privileged_nonowner = InteractionContext(
        guild_id=GUILD_ID,
        user_id=SCAM_UPLOADER + 5,
        member_permissions=int(Permission.ADMINISTRATOR),
        command="",
    )
    resp = await handle_component(privileged_nonowner, ComponentAction.APPEAL_OPEN, det.id, deps)
    assert resp.i18n_key == "command.appeal_none"
    assert (await db_session.execute(Appeal.__table__.select())).fetchall() == []


# --- resilience: circuit breaker + rate limiting ----------------------------


async def test_open_circuit_breaker_fails_action_gracefully(
    db_session: _Session, redis: fakeredis.aioredis.FakeRedis
) -> None:
    # A breaker that trips after a single failure, on a fake clock so it never
    # auto-recovers mid-test. The first scam trips it; the second is rejected
    # fast with circuit_open and no Discord call is attempted.
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_time=1000.0, time_source=clock.now)
    scam = make_scam_png()
    await _seed_guild(db_session, action_policy="delete_ban")
    await _register_scam_hash(db_session, scam)

    # REST fails the delete, which fails the action and trips the breaker open.
    rest = RecordingRest(fail_on=frozenset({"delete_message"}))
    bus = await _wire_pipeline(db_session, redis, rest=rest, breaker=breaker)

    await bus.publish(
        SUBJECT_IMAGE_FETCHED,
        image_fetched_event(
            scam, guild_id=GUILD_ID, uploader_id=SCAM_UPLOADER, idempotency_key="cb-1"
        ),
    )
    first = bus.events(SUBJECT_ACTION_RESULT)[0]
    assert isinstance(first, ActionResultEvent)
    assert first.success is False
    assert first.detail is not None and first.detail.startswith("error:")

    from optimus.core.circuit import CircuitState

    assert breaker.state is CircuitState.OPEN
    calls_after_first = len(rest.calls)

    # A second scam (different uploader/key) is now short-circuited: the breaker
    # is open, so the executor returns circuit_open without touching Discord.
    await bus.publish(
        SUBJECT_IMAGE_FETCHED,
        image_fetched_event(
            scam,
            guild_id=GUILD_ID,
            uploader_id=SCAM_UPLOADER + 1,
            idempotency_key="cb-2",
            message_id=556,
        ),
    )
    second = bus.events(SUBJECT_ACTION_RESULT)[1]
    assert isinstance(second, ActionResultEvent)
    assert second.success is False
    assert second.detail == "circuit_open"
    # No new Discord REST calls were attempted while the circuit was open.
    assert len(rest.calls) == calls_after_first
    # Both failures are still audited as attempts.
    audits = await ModActionRepository(db_session, GUILD_ID).list_recent()
    assert sum(1 for a in audits if a.action == "delete_ban") == 2


async def test_rate_limited_action_fails_without_discord_call(
    db_session: _Session, redis: fakeredis.aioredis.FakeRedis
) -> None:
    # A token bucket with capacity 1 lets the first action through and starves
    # the second within the same instant (refill is far slower than the test).
    scam = make_scam_png()
    await _seed_guild(db_session, action_policy="delete_ban")
    await _register_scam_hash(db_session, scam)

    rest = RecordingRest()
    bus = await _wire_pipeline(
        db_session, redis, rest=rest, rate=RateLimit(capacity=1.0, refill_rate=0.0001)
    )

    await bus.publish(
        SUBJECT_IMAGE_FETCHED,
        image_fetched_event(
            scam, guild_id=GUILD_ID, uploader_id=SCAM_UPLOADER, idempotency_key="rl-1"
        ),
    )
    first = bus.events(SUBJECT_ACTION_RESULT)[0]
    assert isinstance(first, ActionResultEvent)
    assert first.success is True  # first token consumed, ban applied
    assert "ban_member" in rest.verbs
    calls_after_first = len(rest.calls)

    await bus.publish(
        SUBJECT_IMAGE_FETCHED,
        image_fetched_event(
            scam,
            guild_id=GUILD_ID,
            uploader_id=SCAM_UPLOADER + 1,
            idempotency_key="rl-2",
            message_id=557,
        ),
    )
    second = bus.events(SUBJECT_ACTION_RESULT)[1]
    assert isinstance(second, ActionResultEvent)
    assert second.success is False
    assert second.detail == "rate_limited"
    # The starved action makes no Discord call at all.
    assert len(rest.calls) == calls_after_first


class _FakeClock:
    """A frozen monotonic clock so the breaker never auto-recovers mid-test."""

    def now(self) -> float:
        return 0.0
