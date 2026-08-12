"""Tests for the SIGTERM/SIGINT shutdown handler added to ``run_simple``.

Railway (and most container platforms) send SIGTERM to stop or redeploy a
service. Python's default action for an unhandled SIGTERM is immediate process
termination — no ``finally`` blocks run. Before this fix, ``run_simple`` had no
signal handling at all, so a redeploy could kill the process mid-flight without
ever reaching ``app.aclose()`` (and therefore ``engine.dispose()``), leaving the
SQLite file on the shared Railway volume with a connection that was never
cleanly released. The next container starting up could then hit "database is
locked" errors that look transient but are actually caused by the previous
process being killed uncleanly.

``_install_shutdown_signal_handler`` closes that gap by registering a real
``asyncio`` signal handler that cancels the serving task, so the existing
``finally`` chain (which already calls ``app.aclose()``) gets to run.
"""

from __future__ import annotations

import asyncio
import signal

import pytest

from optimus.app.simple import _install_shutdown_signal_handler


async def test_sigterm_cancels_the_serving_task() -> None:
    """Delivering a real SIGTERM to this process cancels the installed task.

    Uses ``os.kill(os.getpid(), signal.SIGTERM)`` rather than calling the
    registered callback directly, so the test exercises the actual OS signal
    delivery path through the running event loop, not just the callback logic.
    """
    import os

    async def _serve_forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.ensure_future(_serve_forever())
    _install_shutdown_signal_handler(task)

    # Let the task actually start running before signalling it.
    await asyncio.sleep(0)

    os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    assert task.cancelled()


async def test_sigint_also_cancels_the_serving_task() -> None:
    """SIGINT (e.g. a local Ctrl+C, or some platforms' equivalent of SIGTERM)
    is handled the same way as SIGTERM.
    """
    import os

    async def _serve_forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.ensure_future(_serve_forever())
    _install_shutdown_signal_handler(task)

    await asyncio.sleep(0)

    os.kill(os.getpid(), signal.SIGINT)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    assert task.cancelled()


async def test_finally_block_runs_before_cancellation_propagates() -> None:
    """The whole point of the fix: cancelling via signal must still let a
    ``finally`` block (standing in for ``app.aclose()``/``engine.dispose()``)
    execute before the task is considered done.
    """
    import os

    cleanup_ran = False

    async def _serve_with_cleanup() -> None:
        nonlocal cleanup_ran
        try:
            await asyncio.Event().wait()
        finally:
            # Simulates app.aclose() -> engine.dispose() in run_simple().
            cleanup_ran = True

    task = asyncio.ensure_future(_serve_with_cleanup())
    _install_shutdown_signal_handler(task)

    await asyncio.sleep(0)

    os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    assert cleanup_ran, "finally block must run before the task finishes on SIGTERM"
