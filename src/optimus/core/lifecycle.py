"""Graceful-shutdown signal wiring shared by the long-lived service mains.

Each distributed service runs its consume loop until a ``stop`` event is set. On
a container rollout the orchestrator sends ``SIGTERM`` and only escalates to
``SIGKILL`` after a grace period; without a ``SIGTERM`` handler the process
ignores it and is hard-killed mid-message, dropping in-flight work a clean drain
would have finished. This installs handlers for ``SIGTERM`` and ``SIGINT`` that
set ``stop`` so the main's ``finally`` drains normally instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

from optimus.core.logging import get_logger

_log = get_logger(__name__)


def install_signal_handlers(stop: asyncio.Event) -> None:
    """Set ``stop`` on SIGTERM/SIGINT so the caller's drain path runs.

    Best-effort: on a platform or loop without ``add_signal_handler`` support
    the call is a no-op and the process keeps its default signal disposition.
    """
    loop = asyncio.get_running_loop()

    def _request_stop(sig: signal.Signals) -> None:
        _log.info("shutdown_signal", signal=sig.name)
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop, sig)
