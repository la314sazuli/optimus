"""Unit tests for the moderation coordinator orchestration."""

from __future__ import annotations

from datetime import UTC, datetime

import fakeredis.aioredis

from optimus.contracts.events import Action, Verdict, VerdictEvent
from optimus.core.backoff import BackoffPolicy
from optimus.core.circuit import CircuitBreaker
from optimus.core.ratelimit import InMemoryRateLimiter, RateLimit
from optimus.services.moderation.actions import ActionExecutor
from optimus.services.moderation.boundaries import TargetContext
from optimus.services.moderation.cooldown import Cooldown
from optimus.services.moderation.coordinator import GuildModConfig, ModerationCoordinator
from optimus.services.moderation.review import ReportData


class _FakeRest:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.dms: list[int] = []

    async def delete_message(self, channel_id: int, message_id: int) -> None:
        self.calls.append("delete_message")

    async def timeout_member(self, guild_id: int, user_id: int, seconds: int) -> None:
        self.calls.append("timeout_member")

    async def kick_member(self, guild_id: int, user_id: int, reason: str) -> None:
        self.calls.append("kick_member")

    async def ban_member(self, guild_id: int, user_id: int, reason: str) -> None:
        self.calls.append("ban_member")

    async def unban_member(self, guild_id: int, user_id: int, reason: str) -> None:
        self.calls.append("unban_member")

    async def send_dm(self, user_id: int, content: str) -> None:
        self.dms.append(user_id)


def _event(*, verdict: Verdict = Verdict.SCAM, confidence: float = 0.95) -> VerdictEvent:
    return VerdictEvent(
        correlation_id="c",
        occurred_at=datetime.now(UTC),
        guild_id=1,
        channel_id=2,
        message_id=3,
        attachment_id=4,
        uploader_id=42,
        idempotency_key="idem-1",
        verdict=verdict,
        confidence=confidence,
    )


def _build(
    *,
    rest: _FakeRest,
    redis: object,
    cfg: GuildModConfig,
    target: TargetContext | None,
    reports: list[ReportData],
    audits: list[tuple[str, bool]],
) -> ModerationCoordinator:
    from optimus.services.moderation.service import _ActionIdempotency

    executor = ActionExecutor(
        rest,
        InMemoryRateLimiter(),
        bot_user_id=999,
        rate=RateLimit(capacity=10.0, refill_rate=0.001),
        idempotency_acquire=_ActionIdempotency(redis).acquire,
        dm_cooldown=Cooldown(redis, window_seconds=3600),
        breaker=CircuitBreaker(),
        backoff=BackoffPolicy(base=0.001, max_delay=0.002, max_attempts=2),
    )

    async def config(_gid: int) -> GuildModConfig:
        return cfg

    async def resolve_target(_gid: int, _uid: int) -> TargetContext | None:
        return target

    async def post(_chan: int, data: ReportData) -> int | None:
        reports.append(data)
        return 7

    async def audit(event: VerdictEvent, action: str, result: object) -> int | None:
        audits.append((action, result.success))  # type: ignore[attr-defined]
        return 7

    return ModerationCoordinator(
        config=config, target=resolve_target, executor=executor, report=post, audit=audit
    )


def _cfg(**kw: object) -> GuildModConfig:
    base: dict[str, object] = {
        "guild_id": 1,
        "configured_action": Action.DELETE_BAN,
        "mod_queue_threshold": 0.5,
        "auto_act_threshold": 0.85,
        "safe_mode": False,
        "review_channel_id": 100,
    }
    base.update(kw)
    return GuildModConfig(**base)  # type: ignore[arg-type]


def _target(**kw: object) -> TargetContext:
    base: dict[str, object] = {
        "user_id": 42,
        "guild_owner_id": 1,
        "bot_user_id": 999,
        "is_administrator": False,
        "top_role_position": 1,
        "bot_top_role_position": 5,
    }
    base.update(kw)
    return TargetContext(**base)  # type: ignore[arg-type]


async def test_clean_verdict_short_circuits() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    rest = _FakeRest()
    reports: list[ReportData] = []
    audits: list[tuple[str, bool]] = []
    coord = _build(
        rest=rest, redis=redis, cfg=_cfg(), target=_target(), reports=reports, audits=audits
    )
    result = await coord.handle_verdict(_event(verdict=Verdict.CLEAN))
    assert result.action is Action.NONE
    assert rest.calls == []
    assert audits == []


async def test_auto_act_ban_executes_and_audits_and_reports() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    rest = _FakeRest()
    reports: list[ReportData] = []
    audits: list[tuple[str, bool]] = []
    coord = _build(
        rest=rest, redis=redis, cfg=_cfg(), target=_target(), reports=reports, audits=audits
    )
    result = await coord.handle_verdict(_event())
    assert result.success
    assert "ban_member" in rest.calls
    assert audits == [("delete_ban", True)]
    assert len(reports) == 1
    assert reports[0].action_taken == "delete_ban"


async def test_boundary_refusal_downgrades_to_report() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    rest = _FakeRest()
    reports: list[ReportData] = []
    audits: list[tuple[str, bool]] = []
    # Target is the guild owner -> punitive action refused.
    coord = _build(
        rest=rest,
        redis=redis,
        cfg=_cfg(),
        target=_target(user_id=42, guild_owner_id=42),
        reports=reports,
        audits=audits,
    )
    result = await coord.handle_verdict(_event())
    assert result.action is Action.REPORT_ONLY
    assert "ban_member" not in rest.calls
    assert audits == [("report_only", True)]


async def test_missing_member_downgrades_to_report() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    rest = _FakeRest()
    reports: list[ReportData] = []
    audits: list[tuple[str, bool]] = []
    coord = _build(rest=rest, redis=redis, cfg=_cfg(), target=None, reports=reports, audits=audits)
    result = await coord.handle_verdict(_event())
    assert result.action is Action.REPORT_ONLY


async def test_queued_verdict_reports_without_action() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    rest = _FakeRest()
    reports: list[ReportData] = []
    audits: list[tuple[str, bool]] = []
    coord = _build(
        rest=rest, redis=redis, cfg=_cfg(), target=_target(), reports=reports, audits=audits
    )
    result = await coord.handle_verdict(_event(confidence=0.6))
    assert result.action is Action.REPORT_ONLY
    assert rest.calls == []
    assert audits == [("report_only", True)]
    assert len(reports) == 1


async def test_safe_mode_blocks_auto_act() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    rest = _FakeRest()
    reports: list[ReportData] = []
    audits: list[tuple[str, bool]] = []
    coord = _build(
        rest=rest,
        redis=redis,
        cfg=_cfg(safe_mode=True),
        target=_target(),
        reports=reports,
        audits=audits,
    )
    result = await coord.handle_verdict(_event())
    assert result.action is Action.REPORT_ONLY
    assert "ban_member" not in rest.calls
