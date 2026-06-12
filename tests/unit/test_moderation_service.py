"""Tests for the moderation service runtime: verdict handling, guild join, wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import fakeredis.aioredis
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession as _Session

from optimus.contracts.events import (
    SUBJECT_ACTION_RESULT,
    Action,
    ActionResultEvent,
    GuildJoinedEvent,
    SwarmAlertEvent,
    Verdict,
    VerdictEvent,
)
from optimus.core.config import get_settings
from optimus.db.engine import SessionScope, create_engine, create_session_factory, session_scope
from optimus.db.models import Base, Detection, Guild, ModAction
from optimus.services.moderation.actions import ActionResult
from optimus.services.moderation.service import (
    ModerationService,
    build_coordinator,
)


@pytest_asyncio.fixture
async def scope() -> AsyncIterator[SessionScope]:
    engine: AsyncEngine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)

    @asynccontextmanager
    async def _scope() -> AsyncIterator[_Session]:
        async with session_scope(factory) as s:
            yield s

    yield _scope
    await engine.dispose()


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []
        self.msg_ids: list[str | None] = []

    async def publish(self, subject: str, event: object, *, msg_id: str | None = None) -> None:
        self.published.append((subject, event))
        self.msg_ids.append(msg_id)


class _FakeCoordinator:
    def __init__(self, result: ActionResult) -> None:
        self._result = result
        self.seen: list[VerdictEvent] = []

    async def handle_verdict(self, event: VerdictEvent) -> ActionResult:
        self.seen.append(event)
        return self._result


def _verdict() -> VerdictEvent:
    return VerdictEvent(
        correlation_id="corr-1",
        occurred_at=datetime.now(UTC),
        guild_id=7,
        channel_id=2,
        message_id=3,
        attachment_id=4,
        uploader_id=42,
        idempotency_key="idem-7",
        verdict=Verdict.SCAM,
        confidence=0.6,
    )


async def test_on_verdict_runs_coordinator_and_emits_result(scope: SessionScope) -> None:
    bus = _FakeBus()
    coord = _FakeCoordinator(ActionResult(Action.DELETE_BAN, success=True, detail=None))
    svc = ModerationService(get_settings(), bus, coord, scope)  # type: ignore[arg-type]

    event = _verdict()
    await svc.on_verdict(event)

    assert coord.seen == [event]
    assert len(bus.published) == 1
    subject, emitted = bus.published[0]
    assert subject == SUBJECT_ACTION_RESULT
    assert isinstance(emitted, ActionResultEvent)
    # The emitted result mirrors the coordinator outcome and the source event.
    assert emitted.action is Action.DELETE_BAN
    assert emitted.success is True
    assert emitted.correlation_id == "corr-1"
    assert emitted.guild_id == 7
    assert emitted.idempotency_key == "idem-7"


async def test_on_guild_joined_inserts_row_once(scope: SessionScope) -> None:
    svc = ModerationService(
        get_settings(), _FakeBus(), _FakeCoordinator(ActionResult(Action.NONE, True)), scope
    )  # type: ignore[arg-type]
    event = GuildJoinedEvent(correlation_id="c", occurred_at=datetime.now(UTC), guild_id=99)
    await svc.on_guild_joined(event)
    await svc.on_guild_joined(event)  # idempotent: existing row is left in place

    async with scope() as s:
        rows = (await s.execute(Guild.__table__.select())).fetchall()
    assert len(rows) == 1
    assert rows[0].guild_id == 99


async def test_on_swarm_alert_is_observational_noop(scope: SessionScope) -> None:
    bus = _FakeBus()
    svc = ModerationService(
        get_settings(), bus, _FakeCoordinator(ActionResult(Action.NONE, True)), scope
    )  # type: ignore[arg-type]
    await svc.on_swarm_alert(
        SwarmAlertEvent(
            correlation_id="c",
            occurred_at=datetime.now(UTC),
            phash=12345,
            distinct_guilds=3,
            window_seconds=60,
        )
    )
    # A swarm alert neither persists nor emits anything in this service.
    assert bus.published == []


async def test_build_coordinator_config_and_audit_persist_detection(
    scope: SessionScope,
) -> None:
    # Guild configured to ban, but the verdict confidence (0.6) sits between the
    # mod-queue bar (0.5) and the auto-act bar (0.85) -> MOD_QUEUE. That path
    # exercises the config + audit closures without touching the rate limiter
    # (which needs Redis EVAL) or the target/report REST closures.
    async with scope() as s:
        s.add(Guild(guild_id=7, action_policy="delete_ban", mod_queue_threshold=0.5))

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    coordinator, _dispatcher = build_coordinator(
        get_settings(), scope, rest=object(), redis=redis, bot_user_id=999
    )

    result = await coordinator.handle_verdict(_verdict())
    # MOD_QUEUE downgrades to a queued report_only outcome.
    assert result.action is Action.REPORT_ONLY
    assert result.detail == "queued"

    async with scope() as s:
        detections = (await s.execute(Detection.__table__.select())).fetchall()
        mod_actions = (await s.execute(ModAction.__table__.select())).fetchall()
    assert len(detections) == 1
    assert detections[0].idempotency_key == "idem-7"
    assert detections[0].action_taken == "report_only"
    assert len(mod_actions) == 1
    assert mod_actions[0].action == "report_only"
    assert mod_actions[0].target == "42"
    await redis.aclose()


async def test_build_coordinator_config_defaults_for_unconfigured_guild(
    scope: SessionScope,
) -> None:
    # No guild row -> config falls back to report_only / default thresholds, so a
    # high-confidence scam is still only queued, and the detection is recorded.
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    coordinator, _dispatcher = build_coordinator(
        get_settings(), scope, rest=object(), redis=redis, bot_user_id=999
    )
    event = VerdictEvent(
        correlation_id="c",
        occurred_at=datetime.now(UTC),
        guild_id=123,
        channel_id=2,
        message_id=3,
        attachment_id=4,
        uploader_id=42,
        idempotency_key="idem-x",
        verdict=Verdict.SCAM,
        confidence=0.99,
    )
    result = await coordinator.handle_verdict(event)
    assert result.action is Action.REPORT_ONLY  # unconfigured guild never auto-acts
    async with scope() as s:
        detections = (await s.execute(Detection.__table__.select())).fetchall()
    assert len(detections) == 1
    await redis.aclose()


async def test_build_coordinator_wires_circuit_settings(scope: SessionScope) -> None:
    # The breaker defaults must match documented behavior, and overrides must flow
    # through from Settings into the executor's breaker.
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)

    default_settings = get_settings()
    coordinator, _dispatcher = build_coordinator(
        default_settings, scope, rest=object(), redis=redis, bot_user_id=1
    )
    breaker = coordinator._executor._breaker  # type: ignore[attr-defined]
    assert breaker._failure_threshold == default_settings.mod_circuit_failure_threshold
    assert breaker._recovery_time == default_settings.mod_circuit_recovery_seconds
    # Defaults are preserved exactly (the previously-unwired ActionExecutor used 5/30).
    assert breaker._failure_threshold == 5
    assert breaker._recovery_time == 30.0

    custom = default_settings.model_copy(
        update={"mod_circuit_failure_threshold": 2, "mod_circuit_recovery_seconds": 7.5}
    )
    coordinator2, _dispatcher2 = build_coordinator(
        custom, scope, rest=object(), redis=redis, bot_user_id=1
    )
    breaker2 = coordinator2._executor._breaker  # type: ignore[attr-defined]
    assert breaker2._failure_threshold == 2
    assert breaker2._recovery_time == 7.5
    await redis.aclose()


def test_action_idempotency_guard_acquires_once() -> None:
    import asyncio

    from optimus.services.moderation.service import _ActionIdempotency

    async def run() -> None:
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        guard = _ActionIdempotency(redis)
        assert await guard.acquire("k") is True
        assert await guard.acquire("k") is False  # second claim within TTL is denied
        await redis.aclose()

    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
