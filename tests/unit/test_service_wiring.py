"""Regression tests: production builders must carry configured settings.

These guard against the class of defect where a new setting is implemented and
unit-tested in isolation but never passed to the production constructor (so it
defaults to the "off" value in every running service). Each test builds the
real service/worker/limiter from ``Settings`` and asserts the constructed object
reflects both the default and an overridden value. Same pattern as
``tests/unit/test_moderation_service.py::test_build_coordinator_wires_circuit_settings``.
"""

from __future__ import annotations

from typing import cast

from optimus.bus.nats import EventBus
from optimus.core.config import get_settings
from optimus.core.ratelimit import InMemoryRateLimiter
from optimus.services.detection.service import build_service
from optimus.services.ingest.service import build_worker
from optimus.services.interactions.service import build_rate_limiter


def test_build_service_wires_guild_index_cap() -> None:
    settings = get_settings()
    bus = cast(EventBus, object())

    service = build_service(settings, bus, redis=None)
    assert service._indexes._max_guilds == settings.detection_guild_index_cap
    assert service._indexes._max_guilds == 1024  # documented default

    custom = settings.model_copy(update={"detection_guild_index_cap": 7})
    service2 = build_service(custom, bus, redis=None)
    assert service2._indexes._max_guilds == 7


def test_build_worker_wires_ingest_inmemory_sweep() -> None:
    # No Redis => in-memory fallback limiter, which must self-bound via sweep.
    settings = get_settings()

    worker = build_worker(settings, redis=None)
    limiter = worker._limiter
    assert isinstance(limiter, InMemoryRateLimiter)
    assert limiter.sweep_interval == settings.ingest_inmemory_sweep_seconds
    assert limiter.sweep_interval == 300.0  # documented default

    custom = settings.model_copy(update={"ingest_inmemory_sweep_seconds": 42.0})
    limiter2 = build_worker(custom, redis=None)._limiter
    assert isinstance(limiter2, InMemoryRateLimiter)
    assert limiter2.sweep_interval == 42.0


def test_build_rate_limiter_wires_interactions_inmemory_sweep() -> None:
    settings = get_settings()

    limiter = build_rate_limiter(settings, redis=None)
    assert isinstance(limiter, InMemoryRateLimiter)
    assert limiter.sweep_interval == settings.interactions_inmemory_sweep_seconds
    assert limiter.sweep_interval == 300.0  # documented default

    custom = settings.model_copy(update={"interactions_inmemory_sweep_seconds": 99.0})
    limiter2 = build_rate_limiter(custom, redis=None)
    assert isinstance(limiter2, InMemoryRateLimiter)
    assert limiter2.sweep_interval == 99.0
