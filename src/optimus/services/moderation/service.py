"""Moderation service runtime: bus + Discord REST + persistence wiring.

Consumes ``verdict.v1`` (and ``swarm_alert.v1`` for safe-mode signals and
report enrichment), runs the :class:`ModerationCoordinator`, applies actions via
the hikari REST client, posts reports to the auto-provisioned review channel,
and emits ``action_result.v1``. ``guild_joined.v1`` triggers review-channel
provisioning.

Almost everything here is side-effecting glue; the testable logic lives in
:mod:`policy`, :mod:`boundaries`, :mod:`safemode`, :mod:`actions`,
:mod:`coordinator`, and :mod:`review`.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

from optimus.bus import Bus
from optimus.bus.nats import EventBus
from optimus.contracts.events import (
    SUBJECT_ACTION_RESULT,
    SUBJECT_GUILD_JOINED,
    SUBJECT_SWARM_ALERT,
    SUBJECT_VERDICT,
    Action,
    ActionResultEvent,
    GuildJoinedEvent,
    SwarmAlertEvent,
    VerdictEvent,
)
from optimus.core.circuit import CircuitBreaker
from optimus.core.config import Settings, get_settings
from optimus.core.health import HealthServer
from optimus.core.lifecycle import install_signal_handlers
from optimus.core.logging import configure_logging, get_logger
from optimus.core.ratelimit import InMemoryRateLimiter, RateLimit, RateLimiter, RedisRateLimiter
from optimus.core.readiness import db_check, nats_check, redis_check
from optimus.db.engine import (
    SessionScope,
    create_engine,
    create_session_factory,
    create_session_scope,
)
from optimus.db.models import Detection, Guild
from optimus.db.repositories import (
    DetectionRepository,
    GuildRepository,
    ModActionRepository,
)
from optimus.services.moderation.actions import ActionExecutor, ActionResult
from optimus.services.moderation.boundaries import TargetContext
from optimus.services.moderation.cooldown import Cooldown
from optimus.services.moderation.coordinator import GuildModConfig, ModerationCoordinator
from optimus.services.moderation.priority import PriorityDispatcher
from optimus.services.moderation.review import ReportData

_log = get_logger(__name__)

#: Audit actor id used when the system (not a human moderator) acts.
SYSTEM_ACTOR = 0


class ModerationService:
    """Owns the coordinator, persistence, and bus emission for moderation."""

    def __init__(
        self,
        settings: Settings,
        bus: Bus,
        coordinator: ModerationCoordinator,
        session_scope_factory: SessionScope,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._coordinator = coordinator
        self._scope = session_scope_factory

    async def on_verdict(self, event: VerdictEvent) -> None:
        """Decide + apply moderation for one verdict and emit the result."""
        result = await self._coordinator.handle_verdict(event)
        await self._emit_result(event, result)

    async def on_swarm_alert(self, event: SwarmAlertEvent) -> None:
        """Record a swarm alert (safe-mode signals are evaluated upstream)."""
        _log.info(
            "swarm_alert_received",
            distinct_guilds=event.distinct_guilds,
            window_seconds=event.window_seconds,
        )

    async def on_guild_joined(self, event: GuildJoinedEvent) -> None:
        """Ensure the guild row exists so config + review channel can be set up."""
        async with self._scope(event.guild_id) as session:
            repo = GuildRepository(session)
            if await repo.get(event.guild_id) is None:
                await repo.upsert(Guild(guild_id=event.guild_id))
        _log.info("guild_joined", guild_id=event.guild_id)

    async def _emit_result(self, event: VerdictEvent, result: ActionResult) -> None:
        await self._bus.publish(
            SUBJECT_ACTION_RESULT,
            ActionResultEvent(
                correlation_id=event.correlation_id,
                occurred_at=datetime.now(UTC),
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                message_id=event.message_id,
                attachment_id=event.attachment_id,
                uploader_id=event.uploader_id,
                idempotency_key=event.idempotency_key,
                action=result.action,
                success=result.success,
                detail=result.detail,
            ),
            msg_id=f"{event.idempotency_key}:{result.action.value}",
        )


def build_coordinator(
    settings: Settings,
    scope: SessionScope,
    *,
    rest: object,
    redis: object,
    bot_user_id: int,
    rate_limiter: RateLimiter | None = None,
) -> tuple[ModerationCoordinator, PriorityDispatcher[ActionResult]]:
    """Wire a :class:`ModerationCoordinator` from settings and shared clients.

    Returns the coordinator and its :class:`PriorityDispatcher`; the caller owns
    the dispatcher lifecycle (``start``/``stop``).

    ``rate_limiter`` is injectable so simple mode can supply the process-local
    :class:`InMemoryRateLimiter` (no Redis Lua ``EVAL``); when omitted the
    distributed default is the shared :class:`RedisRateLimiter`, unchanged.
    """
    # A runtime Redis outage degrades to a per-process per-guild bucket rather
    # than crashing the action path; the bucket count is bounded by guild count.
    if rate_limiter is None:
        rate_limiter = RedisRateLimiter(
            redis,
            prefix=settings.ratelimit_redis_prefix,
            fallback=InMemoryRateLimiter(),
        )
    cooldown = Cooldown(redis, window_seconds=settings.mod_dm_cooldown_seconds)
    guard = _ActionIdempotency(redis)

    breaker = CircuitBreaker(
        failure_threshold=settings.mod_circuit_failure_threshold,
        recovery_time=settings.mod_circuit_recovery_seconds,
    )
    executor = ActionExecutor(
        rest,  # type: ignore[arg-type]
        rate_limiter,
        bot_user_id=bot_user_id,
        rate=RateLimit(
            capacity=settings.mod_action_rate_capacity,
            refill_rate=settings.mod_action_rate_refill,
        ),
        idempotency_acquire=guard.acquire,
        idempotency_release=guard.release,
        dm_cooldown=cooldown,
        breaker=breaker,
    )

    dispatcher: PriorityDispatcher[ActionResult] = PriorityDispatcher(
        concurrency=settings.mod_dispatch_concurrency,
        max_queue=settings.mod_dispatch_max_queue,
        aging_seconds=settings.mod_dispatch_aging_seconds,
    )

    async def config(guild_id: int) -> GuildModConfig:
        async with scope(guild_id) as session:
            guild = await GuildRepository(session).get(guild_id)
            action = Action(guild.action_policy) if guild is not None else Action.REPORT_ONLY
            return GuildModConfig(
                guild_id=guild_id,
                configured_action=action,
                mod_queue_threshold=(
                    guild.mod_queue_threshold if guild is not None else settings.mod_queue_threshold
                ),
                auto_act_threshold=settings.mod_auto_act_threshold,
                safe_mode=guild.safe_mode if guild is not None else False,
                locale=guild.locale if guild is not None else "en",
                review_channel_id=guild.review_channel_id if guild is not None else None,
                timeout_seconds=settings.mod_timeout_seconds,
            )

    async def target(guild_id: int, user_id: int) -> TargetContext | None:  # pragma: no cover
        return await _resolve_target(rest, guild_id, user_id, bot_user_id)

    async def report(channel_id: int, data: ReportData) -> int | None:  # pragma: no cover
        return await _post_report(rest, channel_id, data)

    async def audit(event: VerdictEvent, action: str, result: ActionResult) -> int | None:
        async with scope(event.guild_id) as session:
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
                actor_id=SYSTEM_ACTOR,
                action=action,
                target=str(event.uploader_id),
                payload={"success": result.success, "detail": result.detail},
            )
            return detection.id

    coordinator = ModerationCoordinator(
        config=config,
        target=target,
        executor=executor,
        report=report,
        audit=audit,
        dispatcher=dispatcher,
    )
    return coordinator, dispatcher


class _ActionIdempotency:
    """SET NX guard for action execution (separate keyspace from detection)."""

    def __init__(self, redis: object, *, ttl_seconds: int = 86_400) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    async def acquire(self, key: str) -> bool:
        result = await self._redis.set(key, "1", nx=True, ex=self._ttl)  # type: ignore[attr-defined]
        return result is True or result == "OK"

    async def release(self, key: str) -> None:
        """Release a claim so a failed action's redelivery can re-run."""
        await self._redis.delete(key)  # type: ignore[attr-defined]


async def _resolve_target(  # pragma: no cover - requires live REST
    rest: object, guild_id: int, user_id: int, bot_user_id: int
) -> TargetContext | None:
    import hikari

    try:
        member = await rest.fetch_member(guild_id, user_id)  # type: ignore[attr-defined]
        guild = await rest.fetch_guild(guild_id)  # type: ignore[attr-defined]
        roles = await rest.fetch_roles(guild_id)  # type: ignore[attr-defined]
        me = await rest.fetch_member(guild_id, bot_user_id)  # type: ignore[attr-defined]
    except hikari.NotFoundError:
        return None
    by_id = {int(r.id): r for r in roles}
    is_admin = any(
        (by_id[int(rid)].permissions & hikari.Permissions.ADMINISTRATOR)
        for rid in member.role_ids
        if int(rid) in by_id
    )
    top = max((by_id[int(rid)].position for rid in member.role_ids if int(rid) in by_id), default=0)
    bot_top = max((by_id[int(rid)].position for rid in me.role_ids if int(rid) in by_id), default=0)
    return TargetContext(
        user_id=user_id,
        guild_owner_id=int(guild.owner_id),
        bot_user_id=bot_user_id,
        is_administrator=is_admin,
        top_role_position=top,
        bot_top_role_position=bot_top,
    )


async def _post_report(  # pragma: no cover
    rest: object, channel_id: int, data: ReportData
) -> int | None:
    from optimus.services.moderation.review import build_action_rows, build_embed

    message = await rest.create_message(  # type: ignore[attr-defined]
        channel_id,
        embed=build_embed(data),
        components=build_action_rows(data.detection_id),
    )
    return int(message.id)


async def _amain() -> None:  # pragma: no cover - runtime entrypoint
    settings = get_settings()
    configure_logging(level=settings.log_level, service_name="optimus-moderation")

    bus, nc = await EventBus.connect(settings.nats_url)
    await bus.ensure_stream(duplicate_window=settings.bus_duplicate_window_seconds)

    import redis.asyncio as aioredis

    redis = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=settings.redis_socket_timeout,
        socket_connect_timeout=settings.redis_socket_timeout,
    )

    import hikari

    rest_app = hikari.RESTApp()
    await rest_app.start()
    rest = rest_app.acquire(settings.discord_token, token_type=hikari.TokenType.BOT)
    rest.start()
    me = await rest.fetch_my_user()
    bot_user_id = int(me.id)

    engine = create_engine()
    factory = create_session_factory(engine)
    scope = create_session_scope(factory, multi_tenant=settings.is_multi_tenant)

    coordinator, dispatcher = build_coordinator(
        settings, scope, rest=rest, redis=redis, bot_user_id=bot_user_id
    )
    await dispatcher.start()
    service = ModerationService(settings, bus, coordinator, scope)

    health = HealthServer(host=settings.health_host, port=settings.health_port)
    health.add_readiness_check(nats_check(nc), name="nats")
    # Moderation persists case/action rows; with Postgres down it can only
    # nak-and-redeliver. Gate readiness on the DB so the probe is honest during a
    # DB outage, consistent with detection and interactions.
    health.add_readiness_check(db_check(scope), name="postgres")
    health.add_readiness_check(redis_check(redis), name="redis")
    await health.start()

    stop = asyncio.Event()
    install_signal_handlers(stop)
    tasks = [
        asyncio.create_task(
            bus.consume(
                SUBJECT_VERDICT,
                durable="moderation",
                model=VerdictEvent,
                handler=service.on_verdict,
                batch=settings.detection_fetch_batch,
                max_deliver=settings.detection_max_deliver,
                max_inflight=settings.detection_max_inflight,
                ack_wait=settings.detection_ack_wait_seconds,
                stop_event=stop,
            )
        ),
        asyncio.create_task(
            bus.consume(
                SUBJECT_SWARM_ALERT,
                durable="moderation-swarm",
                model=SwarmAlertEvent,
                handler=service.on_swarm_alert,
                stop_event=stop,
            )
        ),
        asyncio.create_task(
            bus.consume(
                SUBJECT_GUILD_JOINED,
                durable="moderation-join",
                model=GuildJoinedEvent,
                handler=service.on_guild_joined,
                stop_event=stop,
            )
        ),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        health.set_live(False)
        stop.set()
        with contextlib.suppress(Exception):
            await dispatcher.stop()
        with contextlib.suppress(Exception):
            await rest.close()
        with contextlib.suppress(Exception):
            await rest_app.close()
        with contextlib.suppress(Exception):
            await nc.drain()
        await health.stop()
        await engine.dispose()


def main() -> None:  # pragma: no cover
    """Console entrypoint: ``python -m optimus.services.moderation``."""
    asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    main()
