"""Single-process composition: every service in one asyncio process.

``OPTIMUS_MODE=simple`` runs Optimus with **zero external services** — no NATS,
no Redis, no Postgres. The six service runtimes that are normally separate
processes wired over JetStream are instead composed here over the
:class:`~optimus.bus.inprocess.InProcessBus`, sharing one SQLite engine, one
in-memory key/value store (:class:`~optimus.app.memory.MemoryStore`), and one
in-process rate limiter. The detection core and every service's logic are
untouched; this module only *composes* them.

The composition splits cleanly in two:

* The **bus pipeline** — ingest, detection, moderation, and the scheduler — needs
  no live Discord connection. It is wired by :meth:`SimpleApp.build` and is what
  the composition test drives by publishing a synthetic image event.
* The **Discord edges** — the gateway (inbound messages) and interactions
  (slash commands / buttons) — need a real gateway connection and are started by
  :meth:`SimpleApp.run`. The moderation REST client is likewise a real Discord
  edge, built lazily at run time (or injected by a test).

Durability note: the in-process bus keeps queued-but-unprocessed messages only in
memory, so a restart in simple mode loses anything still in flight. Run
``OPTIMUS_MODE=distributed`` when at-least-once durability across restarts
matters.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from functools import partial

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from optimus.app.memory import MemoryStore
from optimus.app.migrate import run_migrations
from optimus.bus.inprocess import InProcessBus
from optimus.contracts.events import (
    SUBJECT_GUILD_JOINED,
    SUBJECT_IMAGE_FETCHED,
    SUBJECT_MESSAGE_IMAGE,
    SUBJECT_SWARM_ALERT,
    SUBJECT_VERDICT,
    GuildJoinedEvent,
    ImageFetchedEvent,
    MessageImageEvent,
    SwarmAlertEvent,
    VerdictEvent,
)
from optimus.core.config import Settings, get_settings
from optimus.core.health import HealthServer
from optimus.core.logging import configure_logging, get_logger
from optimus.core.ratelimit import InMemoryRateLimiter
from optimus.core.readiness import db_check
from optimus.db.engine import SessionScope, create_engine, create_session_factory, session_scope
from optimus.services.detection.service import DetectionService
from optimus.services.detection.service import build_service as build_detection
from optimus.services.ingest.service import _handle as ingest_handle
from optimus.services.ingest.service import build_worker as build_ingest
from optimus.services.ingest.worker import IngestWorker
from optimus.services.moderation.actions import ActionResult
from optimus.services.moderation.priority import PriorityDispatcher
from optimus.services.moderation.service import ModerationService, build_coordinator
from optimus.services.scheduler.service import SchedulerService

_log = get_logger(__name__)


@dataclass
class SimpleApp:
    """A fully wired single-process Optimus, minus the live Discord connection.

    Construct with :meth:`build`. The bus pipeline (ingest -> detection ->
    moderation) and scheduler are ready immediately; :meth:`start_pipeline` runs
    the bus consumers and scheduler loops, and the Discord edges are started
    separately by :func:`optimus.app.discord.run_discord_edges`.
    """

    settings: Settings
    engine: AsyncEngine
    bus: InProcessBus
    store: MemoryStore
    health: HealthServer
    detection: DetectionService
    moderation: ModerationService
    scheduler: SchedulerService
    dispatcher: PriorityDispatcher[ActionResult]
    ingest_worker: IngestWorker
    _scope: SessionScope
    _rest: object
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _consumer_tasks: list[asyncio.Task[None]] = field(default_factory=list)
    _scheduler_tasks: list[asyncio.Task[None]] = field(default_factory=list)

    @classmethod
    async def build(
        cls,
        settings: Settings | None = None,
        *,
        rest: object | None = None,
        bot_user_id: int = 0,
        run_migrations_on_start: bool = True,
    ) -> SimpleApp:
        """Wire every service over the in-process bus, SQLite, and in-memory stores.

        ``rest`` injects the Discord REST surface used by moderation (a real
        hikari client at run time, or a recording double in tests). When omitted a
        no-op REST is used so the pipeline can run without Discord — the
        composition test relies on this. ``run_migrations_on_start`` brings the
        SQLite schema up to date before wiring anything that touches the DB.
        """
        settings = settings or get_settings()
        url = settings.effective_database_url
        if run_migrations_on_start:
            await run_migrations(url)

        engine = create_engine(url, settings=settings)
        factory = create_session_factory(engine)

        def scope() -> AbstractAsyncContextManager[AsyncSession]:
            return session_scope(factory)

        bus = InProcessBus(duplicate_window=settings.bus_duplicate_window_seconds)
        store = MemoryStore()

        ingest_worker = build_ingest(settings, store)
        detection = build_detection(
            settings, bus, store, session_scope_factory=scope, enable_swarm=False
        )

        coordinator, dispatcher = build_coordinator(
            settings,
            scope,
            rest=rest if rest is not None else _NoopRest(),
            redis=store,
            bot_user_id=bot_user_id,
            rate_limiter=InMemoryRateLimiter(),
        )
        moderation = ModerationService(settings, bus, coordinator, scope)

        scheduler = SchedulerService(settings, bus, scope)

        health = HealthServer(host=settings.health_host, port=settings.health_port)
        health.add_readiness_check(db_check(scope), name="database")

        return cls(
            settings=settings,
            engine=engine,
            bus=bus,
            store=store,
            health=health,
            detection=detection,
            moderation=moderation,
            scheduler=scheduler,
            dispatcher=dispatcher,
            ingest_worker=ingest_worker,
            _scope=scope,
            _rest=rest,
        )

    def start_pipeline(self) -> None:
        """Launch the bus consumers and scheduler loops (no Discord required).

        Each consumer is *registered* synchronously (via :meth:`InProcessBus.run`)
        before its delivery loop runs, so an event published immediately after this
        returns is never lost to a task-scheduling race — the in-process bus drops
        a publish to a subject that has no registered consumer yet.
        """
        s = self.settings
        run = self.bus.run
        self._consumer_tasks = [
            run(
                SUBJECT_MESSAGE_IMAGE,
                durable="ingest",
                model=MessageImageEvent,
                handler=partial(ingest_handle, self.ingest_worker, self.bus),
                stop_event=self._stop,
            ),
            run(
                SUBJECT_IMAGE_FETCHED,
                durable="detection",
                model=ImageFetchedEvent,
                handler=self.detection.on_image,
                max_inflight=s.detection_max_inflight,
                max_deliver=s.detection_max_deliver,
                stop_event=self._stop,
            ),
            run(
                SUBJECT_VERDICT,
                durable="moderation",
                model=VerdictEvent,
                handler=self.moderation.on_verdict,
                max_inflight=s.detection_max_inflight,
                max_deliver=s.detection_max_deliver,
                stop_event=self._stop,
            ),
            run(
                SUBJECT_SWARM_ALERT,
                durable="moderation-swarm",
                model=SwarmAlertEvent,
                handler=self.moderation.on_swarm_alert,
                stop_event=self._stop,
            ),
            run(
                SUBJECT_GUILD_JOINED,
                durable="moderation-join",
                model=GuildJoinedEvent,
                handler=self.moderation.on_guild_joined,
                stop_event=self._stop,
            ),
        ]
        self._scheduler_tasks = self.scheduler.start()

    async def aclose(self) -> None:
        """Stop the pipeline, drain tasks, and dispose the engine."""
        self.health.set_live(False)
        self._stop.set()
        self.scheduler.request_stop()
        for task in self._scheduler_tasks:
            task.cancel()
        all_tasks = self._consumer_tasks + self._scheduler_tasks
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await self.dispatcher.stop()
        await self.engine.dispose()


class _NoopRest:
    """A REST double that records nothing and performs no Discord calls.

    Used only when simple mode runs the pipeline without a live Discord
    connection (e.g. the composition test injects its own recording double, and a
    fully wired run replaces this with a real hikari REST client).
    """

    async def delete_message(self, channel_id: int, message_id: int) -> None: ...
    async def timeout_member(self, guild_id: int, user_id: int, seconds: int) -> None: ...
    async def kick_member(self, guild_id: int, user_id: int, reason: str) -> None: ...
    async def ban_member(self, guild_id: int, user_id: int, reason: str) -> None: ...
    async def unban_member(self, guild_id: int, user_id: int, reason: str) -> None: ...
    async def send_dm(self, user_id: int, content: str) -> None: ...


async def run_simple() -> None:  # pragma: no cover - runtime entrypoint
    """Entrypoint for ``OPTIMUS_MODE=simple``: compose and run everything.

    Requires only ``OPTIMUS_DISCORD_TOKEN``. Validates the few first-run mistakes
    up front (see :func:`optimus.app.startup.validate_simple_startup`), brings up
    the SQLite schema, wires every service over the in-process bus, starts the
    health/metrics server, registers slash commands, and connects the Discord
    gateway + interactions. Blocks until interrupted.
    """
    settings = get_settings()
    configure_logging(level=settings.log_level, service_name="optimus")
    _quiet_dependency_logs(settings.log_level)

    from optimus.app.startup import validate_simple_startup

    validate_simple_startup(settings)

    import hikari

    from optimus.app.startup import StartupError

    rest_app = hikari.RESTApp()
    await rest_app.start()
    rest = rest_app.acquire(settings.discord_token, token_type=hikari.TokenType.BOT)
    rest.start()
    # First network use of the token. A rejected token surfaces here as a hikari
    # 401; translate it (and a connectivity failure) into the same one-line
    # StartupError the local checks raise, so the first thing a new user sees is
    # "Discord rejected the token", not a traceback. Tear down what we opened
    # before re-raising, since we have not yet reached the run try/finally.
    try:
        me = await rest.fetch_my_user()
    except hikari.UnauthorizedError as exc:
        await _close_rest(rest, rest_app)
        raise StartupError(
            "Discord rejected the token (401 Unauthorized). Check that "
            "OPTIMUS_DISCORD_TOKEN is the bot token from the Developer Portal's "
            "Bot tab (use Reset Token to get a fresh one) and that it has not been "
            "revoked or regenerated."
        ) from exc
    except hikari.HTTPError as exc:
        await _close_rest(rest, rest_app)
        raise StartupError(
            f"Could not reach Discord to verify the token ({exc}). Check your "
            "network connection and try again."
        ) from exc
    bot_user_id = int(me.id)

    await _register_commands(rest, bot_user_id)

    app = await SimpleApp.build(settings, rest=rest, bot_user_id=bot_user_id)
    await app.dispatcher.start()
    await app.health.start()
    app.start_pipeline()

    _log.info(
        "optimus_online",
        bot=f"{me.username}#{me.discriminator}",
        bot_id=bot_user_id,
        health_url=f"http://{settings.health_host}:{settings.health_port}/healthz",
    )

    from optimus.app.discord import run_discord_edges

    try:
        await run_discord_edges(app, settings, rest=rest)
    finally:
        await app.aclose()
        await app.health.stop()
        await _close_rest(rest, rest_app)


async def _close_rest(rest: object, rest_app: object) -> None:  # pragma: no cover - net
    """Close the REST client and its app, ignoring teardown errors.

    Used both on the auth-failure path (before the run try/finally exists) and at
    normal shutdown, so a half-open token check does not leak an aiohttp session.
    """
    with contextlib.suppress(Exception):
        await rest.close()  # type: ignore[attr-defined]
    with contextlib.suppress(Exception):
        await rest_app.close()  # type: ignore[attr-defined]


async def _register_commands(rest: object, bot_user_id: int) -> None:  # pragma: no cover - net
    """Register slash commands globally using the bot's own application id.

    Simple mode owns one bot, so we register against the application that the
    token already authenticates as — no separate ``OPTIMUS_DISCORD_CLIENT_ID`` or
    manual ``scripts/register_commands.py`` run. Failure here (e.g. a transient
    Discord 5xx) is logged and tolerated: the bot still runs, and commands can be
    registered later with the script.
    """
    import hikari

    from optimus.services.interactions.commands import build_command_builders

    try:
        await rest.set_application_commands(  # type: ignore[attr-defined]
            hikari.Snowflake(bot_user_id),
            build_command_builders(),
        )
    except Exception:
        _log.warning("command_registration_failed", exc_info=True)
        return
    _log.info("commands_registered", scope="global")


def _quiet_dependency_logs(level: str) -> None:
    """Tone down third-party INFO chatter so simple-mode output stays readable.

    hikari logs a wall of gateway/REST INFO lines on every connect. When the
    operator is running at the default INFO level we lift those dependencies to
    WARNING so Optimus's own ``optimus_online`` / detection logs are not buried;
    an explicit ``OPTIMUS_LOG_LEVEL=DEBUG`` opts back into the full firehose.
    """
    import logging

    if level.upper() == "DEBUG":
        return
    for name in ("hikari", "hikari.gateway", "hikari.rest"):
        logging.getLogger(name).setLevel(logging.WARNING)
