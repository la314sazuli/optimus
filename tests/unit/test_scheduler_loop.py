"""Unit tests for the scheduler periodic-loop machinery."""

from __future__ import annotations

import asyncio
import random

import pytest
from sqlalchemy.exc import OperationalError

from optimus.services.scheduler.service import jittered_interval, run_periodic


def test_jittered_interval_within_bounds() -> None:
    rng = random.Random(0)
    for _ in range(100):
        value = jittered_interval(10.0, 0.1, rng)
        assert 10.0 <= value <= 11.0


def test_jittered_interval_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="base"):
        jittered_interval(0.0, 0.1)


async def test_run_periodic_runs_until_stopped() -> None:
    stop = asyncio.Event()
    runs = {"n": 0}

    async def job() -> int:
        runs["n"] += 1
        if runs["n"] >= 3:
            stop.set()
        return 1

    # Tiny interval and zero jitter so the test is fast and deterministic.
    await run_periodic("t", 0.001, job, stop=stop, jitter_fraction=0.0)
    assert runs["n"] >= 3


async def test_run_periodic_isolates_failures() -> None:
    stop = asyncio.Event()
    runs = {"n": 0}

    async def job() -> int:
        runs["n"] += 1
        if runs["n"] == 1:
            raise RuntimeError("boom")
        stop.set()
        return 0

    # A raising run must not kill the loop; the second run still happens.
    await run_periodic("t", 0.001, job, stop=stop, jitter_fraction=0.0)
    assert runs["n"] >= 2


async def test_run_periodic_exits_immediately_when_prestopped() -> None:
    stop = asyncio.Event()
    stop.set()
    runs = {"n": 0}

    async def job() -> int:
        runs["n"] += 1
        return 0

    await run_periodic("t", 100.0, job, stop=stop, jitter_fraction=0.0)
    assert runs["n"] == 0


async def test_run_periodic_retries_a_transient_sqlite_lock() -> None:
    stop = asyncio.Event()
    runs = {"n": 0}

    async def job() -> int:
        runs["n"] += 1
        if runs["n"] == 1:
            raise OperationalError("DELETE", {}, Exception("database is locked"))
        stop.set()
        return 0

    await run_periodic("t", 0.001, job, stop=stop, jitter_fraction=0.0)
    assert runs["n"] == 2
