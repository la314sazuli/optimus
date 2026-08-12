"""Prometheus metrics — auto-exposed at /metrics via the default registry.

Only metrics not already defined elsewhere in the codebase live here.
Existing metrics: optimus_detection_verdicts_total (worker.py),
optimus_moderation_actions_total (coordinator.py), bus/idempotency/ratelimit
counters in their respective modules.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from prometheus_client import Counter, Gauge, Histogram

DETECTION_LATENCY = Histogram(
    "optimus_detection_latency_seconds",
    "Wall-clock seconds to persist a detection verdict.",
    ["guild_id"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
DECODE_FAILURES = Counter(
    "optimus_decode_failures_total", "Image decode failures.", ["reason"]
)
DB_LOCK_RETRIES = Counter(
    "optimus_db_lock_retries_total", "DB lock retries.", ["service"]
)
DISCORD_API_ERRORS = Counter(
    "optimus_discord_api_errors_total", "Discord API errors.", ["error_type"]
)
OUTBOX_LAG = Gauge("optimus_outbox_lag", "Pending outbox messages.")
ACTIVE_GUILDS = Gauge("optimus_active_guilds", "Active guilds served.")


@contextlib.contextmanager
def _safe() -> Iterator[None]:
    with contextlib.suppress(Exception):
        yield


def record_detection(guild_id: int, latency_s: float) -> None:
    with _safe():
        DETECTION_LATENCY.labels(guild_id=str(guild_id)).observe(latency_s)


def record_decode_failure(reason: str) -> None:
    with _safe():
        DECODE_FAILURES.labels(reason=reason).inc()


def record_db_lock_retry(service: str) -> None:
    with _safe():
        DB_LOCK_RETRIES.labels(service=service).inc()


def record_discord_api_error(error_type: str) -> None:
    with _safe():
        DISCORD_API_ERRORS.labels(error_type=error_type).inc()


def set_outbox_lag(pending: int) -> None:
    with _safe():
        OUTBOX_LAG.set(pending)


def set_active_guilds(count: int) -> None:
    with _safe():
        ACTIVE_GUILDS.set(count)
