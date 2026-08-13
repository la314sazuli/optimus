"""Gateway liveness watchdog: self-heal when hikari's reconnect loop wedges.

Production failure mode this exists for (observed 2026-08-10..13): the
gateway websocket dies and hikari's auto-reconnect gets stuck retrying
("failed to communicate with server, reason was: 'Socket has closed'. Will
retry shortly") for **days** without ever re-establishing a session. The
process itself stays perfectly healthy — the health server answers, the
retention scheduler runs — but no gateway events arrive, so every slash
command times out on the Discord side before the bot ever sees it.

Platform healthchecks do not catch this: Railway only probes the healthcheck
path at deploy time, not continuously, and the process never crashes. The
watchdog closes that gap from inside:

* it samples shard connectivity (``is_alive``/``is_connected``, the same
  predicate as :func:`optimus.core.readiness.shards_check`) every
  ``gateway_watchdog_interval_seconds``;
* while every shard is disconnected it tracks how long the outage has lasted;
* once the outage exceeds ``gateway_stale_exit_seconds`` it marks the health
  server not-live (``/healthz`` -> 503, for operators and external monitors)
  and triggers a **graceful** process shutdown (SIGTERM by default — the same
  path as a Railway redeploy, so ``engine.dispose()`` and friends all run).
  The platform's restart policy then starts a fresh process, which builds a
  brand-new gateway session — exactly the manual "redeploy fixes it" remedy,
  automated.

A reconnect that succeeds at any point resets the outage clock, so ordinary
Discord blips, resumes, and rolling gateway restarts never trigger it. The
watchdog reads only shard state — deliberately not message traffic, which
would false-positive on any quiet server.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from collections.abc import Callable

from optimus.core.health import HealthServer
from optimus.core.logging import get_logger
from optimus.core.readiness import shards_check

_log = get_logger(__name__)


def _default_trigger_shutdown() -> None:  # pragma: no cover - process-level effect
    """Request the same graceful shutdown a platform redeploy would (SIGTERM).

    ``run_simple`` installs a SIGTERM handler that cancels the serve task,
    which runs the full ``finally`` chain (``app.aclose``/``engine.dispose``,
    health stop, REST close). Raising the signal in-process reuses that one
    well-tested shutdown path instead of introducing a second one.
    """
    signal.raise_signal(signal.SIGTERM)


class GatewayWatchdog:
    """Monitors shard connectivity and restarts the process when it stays dead.

    ``bot`` is the :class:`hikari.GatewayBot` (anything exposing ``shards``);
    ``health`` is flipped not-live before shutdown so ``/healthz`` reports the
    real state during the grace window. ``trigger_shutdown`` is injectable for
    tests and defaults to raising SIGTERM in-process.
    """

    def __init__(
        self,
        bot: object,
        health: HealthServer,
        *,
        interval_seconds: float,
        stale_exit_seconds: float,
        trigger_shutdown: Callable[[], None] = _default_trigger_shutdown,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._connected_check = shards_check(bot)
        self._health = health
        self._interval = interval_seconds
        self._stale_exit = stale_exit_seconds
        self._trigger_shutdown = trigger_shutdown
        self._clock = clock
        self._disconnected_since: float | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        """Whether a stale-exit budget is configured (``0`` disables)."""
        return self._stale_exit > 0

    def start(self) -> None:
        """Launch the sampling loop (no-op when disabled)."""
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        """Cancel and drain the sampling loop."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            if await self.check_once():
                return

    async def check_once(self) -> bool:
        """Sample connectivity once; ``True`` when shutdown was triggered.

        Exposed for tests (and callable from any scheduler): one sample of the
        shard predicate plus the stale-clock bookkeeping around it.
        """
        if await self._connected_check():
            if self._disconnected_since is not None:
                _log.info(
                    "gateway_reconnected",
                    outage_seconds=round(self._clock() - self._disconnected_since, 1),
                )
            self._disconnected_since = None
            return False

        now = self._clock()
        if self._disconnected_since is None:
            self._disconnected_since = now
            _log.warning("gateway_disconnected", stale_exit_seconds=self._stale_exit)
            return False

        outage = now - self._disconnected_since
        if outage < self._stale_exit:
            _log.warning(
                "gateway_still_disconnected",
                outage_seconds=round(outage, 1),
                stale_exit_seconds=self._stale_exit,
            )
            return False

        _log.error(
            "gateway_stale_restart",
            outage_seconds=round(outage, 1),
            stale_exit_seconds=self._stale_exit,
        )
        # Fail /healthz first so an external monitor sees *why* the process is
        # about to go away even if shutdown takes a moment.
        self._health.set_live(False)
        self._trigger_shutdown()
        return True
