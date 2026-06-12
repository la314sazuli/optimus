"""A minimal async circuit breaker."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TypeVar

T = TypeVar("T")


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted while the circuit is open."""


class CircuitBreaker:
    """Trips open after ``failure_threshold`` consecutive failures.

    While open, calls fail fast until ``recovery_time`` elapses, after which a
    limited number of trial calls are permitted (half-open). The number of
    concurrent trials is bounded by ``success_threshold``: once that many trials
    are in flight, further calls fail fast just as if the circuit were open.
    Enough successes close the circuit; any failure re-opens it.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_time: float = 30.0,
        success_threshold: int = 1,
        time_source: Callable[[], float] = time.monotonic,
        on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        self._failure_threshold = failure_threshold
        self._recovery_time = recovery_time
        self._success_threshold = success_threshold
        self._now = time_source
        self._listeners: list[Callable[[CircuitState, CircuitState], None]] = []
        if on_state_change is not None:
            self._listeners.append(on_state_change)
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = 0.0
        self._trials_in_flight = 0

    def add_state_listener(
        self, listener: Callable[[CircuitState, CircuitState], None]
    ) -> None:
        """Register ``listener`` to fire on every real state transition.

        Idempotent: re-registering an already-attached listener is a no-op, so a
        caller can attach a shared observer to an injected breaker without risking
        a double-fire if the breaker was already wired with it.
        """
        if listener not in self._listeners:
            self._listeners.append(listener)

    def _set_state(self, new_state: CircuitState) -> None:
        """Transition to ``new_state``, notifying listeners on real changes."""
        previous = self._state
        if previous is new_state:
            return
        self._state = new_state
        for listener in self._listeners:
            listener(previous, new_state)

    @property
    def state(self) -> CircuitState:
        """Return the current state, transitioning open -> half-open if due."""
        if (
            self._state is CircuitState.OPEN
            and self._now() - self._opened_at >= self._recovery_time
        ):
            self._successes = 0
            self._trials_in_flight = 0
            self._set_state(CircuitState.HALF_OPEN)
        return self._state

    def _trip(self) -> None:
        self._opened_at = self._now()
        self._failures = 0
        self._successes = 0
        self._trials_in_flight = 0
        self._set_state(CircuitState.OPEN)

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state is CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self._success_threshold:
                self._failures = 0
                self._successes = 0
                self._trials_in_flight = 0
                self._set_state(CircuitState.CLOSED)
        else:
            self._failures = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        if self._state is CircuitState.HALF_OPEN:
            self._trip()
            return
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._trip()

    def allow(self) -> bool:
        """Whether a call should be attempted right now.

        In half-open the answer also depends on whether a trial permit is
        available; this is a read-only check and does *not* reserve a permit.
        Use :meth:`call` (or :meth:`_reserve` / :meth:`_release`) to actually
        run a guarded trial.
        """
        if self.state is CircuitState.OPEN:
            return False
        if self._state is CircuitState.HALF_OPEN:
            return self._trials_in_flight < self._success_threshold
        return True

    def _reserve(self) -> bool:
        """Atomically claim a slot for one call, or return ``False``.

        Synchronous and non-blocking: it never awaits, so within a single event
        loop the read-test-increment is indivisible and no lock is required.
        In half-open it reserves a bounded trial permit; in closed it is a
        cheap state check that adds no contention to the hot path.
        """
        if self.state is CircuitState.OPEN:
            return False
        if self._state is CircuitState.HALF_OPEN:
            if self._trials_in_flight >= self._success_threshold:
                return False
            self._trials_in_flight += 1
        return True

    def _release(self) -> None:
        """Release a half-open trial permit reserved by :meth:`_reserve`."""
        if self._state is CircuitState.HALF_OPEN and self._trials_in_flight > 0:
            self._trials_in_flight -= 1

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute ``func`` through the breaker.

        The trial permit is reserved synchronously *before* awaiting ``func``
        and released after it settles, so the awaited user call never runs while
        holding a lock and concurrent half-open trials stay within
        ``success_threshold``.
        """
        if not self._reserve():
            raise CircuitOpenError("circuit is open")
        try:
            result = await func()
        except Exception:
            self.record_failure()
            self._release()
            raise
        self.record_success()
        self._release()
        return result
