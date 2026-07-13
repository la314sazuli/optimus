"""Tests for graceful-shutdown signal wiring (SIGTERM/SIGINT -> stop event)."""

from __future__ import annotations

import asyncio
import os
import signal

import pytest

from optimus.core.lifecycle import install_signal_handlers


async def _assert_signal_sets_stop(sig: signal.Signals) -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    install_signal_handlers(stop)
    try:
        os.kill(os.getpid(), sig)
        await asyncio.wait_for(stop.wait(), timeout=1.0)
        assert stop.is_set()
    finally:
        for s in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(s)


async def test_sigterm_triggers_graceful_stop() -> None:
    # A container rollout sends SIGTERM; the handler must set the drain event so
    # the service's finally block runs instead of the process being hard-killed.
    await _assert_signal_sets_stop(signal.SIGTERM)


async def test_sigint_triggers_graceful_stop() -> None:
    await _assert_signal_sets_stop(signal.SIGINT)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
