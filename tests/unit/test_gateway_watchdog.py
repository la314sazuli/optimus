"""Tests for the gateway liveness watchdog.

The watchdog exists for one production failure: hikari's reconnect loop can
wedge indefinitely ("Socket has closed. Will retry shortly", repeating for
days) while the process, health server, and schedulers all stay up -- so no
platform mechanism ever restarts the bot and every slash command times out.
These tests drive :meth:`GatewayWatchdog.check_once` directly with a fake
clock and fake shards, covering the full state machine: healthy, blip,
sustained outage -> shutdown, recovery reset, and the disabled config.
"""

from __future__ import annotations

import pytest

from optimus.core.health import HealthServer
from optimus.services.gateway.watchdog import GatewayWatchdog

pytestmark = pytest.mark.asyncio


class _Shard:
    def __init__(self, *, alive: bool = True, connected: bool = True) -> None:
        self.is_alive = alive
        self.is_connected = connected


class _Bot:
    """Minimal ``bot.shards`` shape accepted by readiness.shards_check."""

    def __init__(self, *shards: _Shard) -> None:
        self.shards = dict(enumerate(shards))


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _watchdog(
    bot: _Bot, *, stale_exit: float = 600.0, clock: _Clock | None = None
) -> tuple[GatewayWatchdog, HealthServer, list[str]]:
    health = HealthServer()
    triggered: list[str] = []
    wd = GatewayWatchdog(
        bot,
        health,
        interval_seconds=30.0,
        stale_exit_seconds=stale_exit,
        trigger_shutdown=lambda: triggered.append("shutdown"),
        clock=clock or _Clock(),
    )
    return wd, health, triggered


async def test_connected_gateway_never_triggers() -> None:
    wd, health, triggered = _watchdog(_Bot(_Shard()))
    for _ in range(10):
        assert await wd.check_once() is False
    assert triggered == []
    assert health._live is True


async def test_short_blip_below_budget_does_not_trigger() -> None:
    clock = _Clock()
    bot = _Bot(_Shard(connected=False))
    wd, health, triggered = _watchdog(bot, stale_exit=600.0, clock=clock)

    assert await wd.check_once() is False  # outage clock starts
    clock.now += 599.0
    assert await wd.check_once() is False  # still inside the budget
    assert triggered == []
    assert health._live is True


async def test_sustained_outage_fails_health_and_triggers_shutdown() -> None:
    clock = _Clock()
    bot = _Bot(_Shard(connected=False))
    wd, health, triggered = _watchdog(bot, stale_exit=600.0, clock=clock)

    assert await wd.check_once() is False
    clock.now += 601.0
    assert await wd.check_once() is True
    assert triggered == ["shutdown"]
    # /healthz must report the real state during the shutdown grace window.
    assert health._live is False


async def test_reconnect_resets_the_outage_clock() -> None:
    clock = _Clock()
    shard = _Shard(connected=False)
    bot = _Bot(shard)
    wd, _health, triggered = _watchdog(bot, stale_exit=600.0, clock=clock)

    assert await wd.check_once() is False
    clock.now += 500.0
    shard.is_connected = True  # gateway recovered on its own (normal hikari path)
    assert await wd.check_once() is False
    shard.is_connected = False  # a *new* outage must start a fresh clock
    clock.now += 500.0
    assert await wd.check_once() is False
    clock.now += 599.0
    assert await wd.check_once() is False  # 599s into the new outage: no trigger
    clock.now += 2.0
    assert await wd.check_once() is True
    assert triggered == ["shutdown"]


async def test_partial_shard_outage_counts_as_disconnected() -> None:
    # shards_check requires *every* shard connected; a replica with one dead
    # shard is not serving that shard's guilds, so the watchdog treats it the
    # same as a full outage rather than averaging it away.
    clock = _Clock()
    bot = _Bot(_Shard(), _Shard(connected=False))
    wd, _health, triggered = _watchdog(bot, stale_exit=60.0, clock=clock)

    assert await wd.check_once() is False
    clock.now += 61.0
    assert await wd.check_once() is True
    assert triggered == ["shutdown"]


async def test_zero_budget_disables_the_watchdog() -> None:
    wd, _health, _triggered = _watchdog(_Bot(_Shard(connected=False)), stale_exit=0.0)
    assert wd.enabled is False
    wd.start()  # must be a no-op rather than scheduling a loop
    assert wd._task is None


async def test_start_and_stop_manage_the_sampling_task() -> None:
    wd, _health, _triggered = _watchdog(_Bot(_Shard()))
    wd.start()
    assert wd._task is not None
    task = wd._task
    wd.start()  # idempotent: no second task replaces the first
    assert wd._task is task
    await wd.stop()
    assert wd._task is None
    await wd.stop()  # idempotent stop
