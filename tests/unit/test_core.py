"""Tests for core utilities: backoff, circuit breaker, rate limiting, idempotency."""

from __future__ import annotations

import asyncio
import random

import pytest

from optimus.core.backoff import BackoffPolicy, retry_async
from optimus.core.circuit import CircuitBreaker, CircuitOpenError, CircuitState
from optimus.core.idempotency import IdempotencyGuard, build_key
from optimus.core.ratelimit import InMemoryRateLimiter, RateLimit, RedisRateLimiter

# --- backoff -------------------------------------------------------------------


def test_backoff_ceiling_is_capped() -> None:
    policy = BackoffPolicy(base=0.1, multiplier=2.0, max_delay=1.0)
    assert policy.ceiling(0) == pytest.approx(0.1)
    assert policy.ceiling(100) == 1.0


def test_backoff_delay_within_full_jitter_bounds() -> None:
    policy = BackoffPolicy(base=0.5, multiplier=2.0, max_delay=10.0)
    rng = random.Random(0)
    for attempt in range(5):
        delay = policy.delay(attempt, rng)
        assert 0.0 <= delay <= policy.ceiling(attempt)


def test_backoff_validation() -> None:
    with pytest.raises(ValueError):
        BackoffPolicy(base=0)
    with pytest.raises(ValueError):
        BackoffPolicy(multiplier=0.5)
    with pytest.raises(ValueError, match="max_delay must be >= base"):
        BackoffPolicy(base=1.0, max_delay=0.5)
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        BackoffPolicy(max_attempts=0)


def test_backoff_ceiling_rejects_negative_attempt() -> None:
    with pytest.raises(ValueError, match="attempt must be >= 0"):
        BackoffPolicy().ceiling(-1)


def test_backoff_delays_yields_one_per_attempt_within_bounds() -> None:
    policy = BackoffPolicy(base=0.1, multiplier=2.0, max_delay=10.0, max_attempts=4)
    rng = random.Random(7)
    delays = list(policy.delays(rng))
    assert len(delays) == 4
    for attempt, d in enumerate(delays):
        assert 0.0 <= d <= policy.ceiling(attempt)


async def test_retry_async_does_not_retry_unlisted_exception() -> None:
    attempts = {"n": 0}

    async def raises_keyerror() -> None:
        attempts["n"] += 1
        raise KeyError("not retryable")

    policy = BackoffPolicy(base=0.001, max_delay=0.001, max_attempts=5)
    with pytest.raises(KeyError):
        await retry_async(raises_keyerror, policy, retry_on=(ValueError,))
    assert attempts["n"] == 1  # raised on first attempt, never retried


async def test_retry_async_succeeds_after_failures() -> None:
    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    policy = BackoffPolicy(base=0.001, max_delay=0.001, max_attempts=5)
    result = await retry_async(flaky, policy)
    assert result == "ok"
    assert attempts["n"] == 3


async def test_retry_async_reraises_after_exhaustion() -> None:
    async def always_fail() -> None:
        raise ValueError("nope")

    policy = BackoffPolicy(base=0.001, max_delay=0.001, max_attempts=2)
    with pytest.raises(ValueError):
        await retry_async(always_fail, policy)


# --- circuit breaker -----------------------------------------------------------


def test_circuit_opens_after_threshold() -> None:
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=3, recovery_time=5.0, time_source=lambda: clock["t"])
    assert cb.state is CircuitState.CLOSED
    for _ in range(3):
        cb.record_failure()
    assert cb.state is CircuitState.OPEN
    assert not cb.allow()


def test_circuit_half_open_then_closes_on_success() -> None:
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=1, recovery_time=5.0, time_source=lambda: clock["t"])
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    clock["t"] = 5.0
    assert cb.state is CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitState.CLOSED


def test_circuit_half_open_reopens_on_failure() -> None:
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=1, recovery_time=5.0, time_source=lambda: clock["t"])
    cb.record_failure()
    clock["t"] = 5.0
    assert cb.state is CircuitState.HALF_OPEN
    cb.record_failure()
    assert cb.state is CircuitState.OPEN


def test_circuit_state_change_callback_observes_transitions() -> None:
    clock = {"t": 0.0}
    seen: list[tuple[CircuitState, CircuitState]] = []
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_time=5.0,
        time_source=lambda: clock["t"],
        on_state_change=lambda prev, cur: seen.append((prev, cur)),
    )
    cb.record_failure()  # closed -> open
    clock["t"] = 5.0
    assert cb.state is CircuitState.HALF_OPEN  # open -> half_open on read
    cb.record_success()  # half_open -> closed
    assert seen == [
        (CircuitState.CLOSED, CircuitState.OPEN),
        (CircuitState.OPEN, CircuitState.HALF_OPEN),
        (CircuitState.HALF_OPEN, CircuitState.CLOSED),
    ]


def test_circuit_state_change_callback_skips_noop_transitions() -> None:
    seen: list[tuple[CircuitState, CircuitState]] = []
    cb = CircuitBreaker(
        failure_threshold=2,
        on_state_change=lambda prev, cur: seen.append((prev, cur)),
    )
    cb.record_failure()  # still closed, below threshold
    cb.record_success()  # still closed
    assert seen == []


def test_circuit_add_state_listener_is_idempotent() -> None:
    seen: list[tuple[CircuitState, CircuitState]] = []

    def listener(prev: CircuitState, cur: CircuitState) -> None:
        seen.append((prev, cur))

    cb = CircuitBreaker(failure_threshold=1, on_state_change=listener)
    cb.add_state_listener(listener)  # already wired -> no double-fire
    cb.record_failure()  # closed -> open
    assert seen == [(CircuitState.CLOSED, CircuitState.OPEN)]


def test_circuit_add_state_listener_attaches_to_bare_breaker() -> None:
    seen: list[tuple[CircuitState, CircuitState]] = []
    cb = CircuitBreaker(failure_threshold=1)
    cb.add_state_listener(lambda prev, cur: seen.append((prev, cur)))
    cb.record_failure()  # closed -> open
    assert seen == [(CircuitState.CLOSED, CircuitState.OPEN)]


async def test_circuit_call_fast_fails_when_open() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_time=100.0)
    cb.record_failure()

    async def fn() -> int:
        return 1

    with pytest.raises(CircuitOpenError):
        await cb.call(fn)


async def test_circuit_half_open_admits_single_trial_concurrently() -> None:
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=1, recovery_time=5.0, time_source=lambda: clock["t"])
    cb.record_failure()
    clock["t"] = 5.0  # recovery elapsed -> half-open on next state read

    gate = asyncio.Event()
    started = {"n": 0}

    async def slow() -> int:
        started["n"] += 1
        await gate.wait()
        return 1

    # The first trial reserves the only permit and parks on the gate; while it
    # is in flight, further calls must fail fast rather than pile on.
    first = asyncio.create_task(cb.call(slow))
    await asyncio.sleep(0)  # let `first` reserve its permit and start awaiting

    for _ in range(5):
        with pytest.raises(CircuitOpenError):
            await cb.call(slow)

    assert started["n"] == 1  # only the permitted trial actually ran
    gate.set()
    assert await first == 1
    assert cb.state is CircuitState.CLOSED  # success closed the circuit


async def test_circuit_half_open_permits_match_success_threshold() -> None:
    clock = {"t": 0.0}
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_time=5.0,
        success_threshold=2,
        time_source=lambda: clock["t"],
    )
    cb.record_failure()
    clock["t"] = 5.0

    gate = asyncio.Event()
    started = {"n": 0}

    async def slow() -> int:
        started["n"] += 1
        await gate.wait()
        return 1

    t1 = asyncio.create_task(cb.call(slow))
    t2 = asyncio.create_task(cb.call(slow))
    await asyncio.sleep(0)

    # Two permits available, so a third concurrent trial is rejected.
    with pytest.raises(CircuitOpenError):
        await cb.call(slow)
    assert started["n"] == 2

    gate.set()
    assert await t1 == 1
    assert await t2 == 1
    assert cb.state is CircuitState.CLOSED


async def test_circuit_half_open_permit_freed_after_failure_reopens() -> None:
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=1, recovery_time=5.0, time_source=lambda: clock["t"])
    cb.record_failure()
    clock["t"] = 5.0
    assert cb.state is CircuitState.HALF_OPEN

    async def boom() -> int:
        raise RuntimeError("trial failed")

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    # The failed trial re-opens the circuit and leaves no leaked permits behind.
    assert cb.state is CircuitState.OPEN
    assert cb._trials_in_flight == 0


async def test_circuit_half_open_reopens_then_recovers_again() -> None:
    clock = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=1, recovery_time=5.0, time_source=lambda: clock["t"])
    cb.record_failure()
    clock["t"] = 5.0

    async def boom() -> int:
        raise RuntimeError("nope")

    async def ok() -> int:
        return 1

    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.state is CircuitState.OPEN

    clock["t"] = 10.0  # second recovery window elapses
    assert cb.state is CircuitState.HALF_OPEN
    assert await cb.call(ok) == 1
    assert cb.state is CircuitState.CLOSED


async def test_circuit_closed_allows_unbounded_concurrency() -> None:
    cb = CircuitBreaker(failure_threshold=3)
    assert cb.state is CircuitState.CLOSED

    async def ok() -> int:
        await asyncio.sleep(0)
        return 1

    results = await asyncio.gather(*(cb.call(ok) for _ in range(50)))
    assert results == [1] * 50
    assert cb.state is CircuitState.CLOSED  # no spurious permit starvation
    assert cb._trials_in_flight == 0


# --- rate limiting -------------------------------------------------------------


async def test_in_memory_token_bucket_limits_and_refills() -> None:
    clock = {"t": 0.0}
    limiter = InMemoryRateLimiter(time_source=lambda: clock["t"])
    limit = RateLimit(capacity=2, refill_rate=1.0)
    assert await limiter.acquire("k", limit)
    assert await limiter.acquire("k", limit)
    assert not await limiter.acquire("k", limit)
    clock["t"] = 1.0  # one token refilled
    assert await limiter.acquire("k", limit)


async def test_rate_limit_keys_are_independent() -> None:
    limiter = InMemoryRateLimiter()
    limit = RateLimit(capacity=1, refill_rate=1.0)
    assert await limiter.acquire("a", limit)
    assert await limiter.acquire("b", limit)


async def test_in_memory_rate_limit_rejects_nonpositive_cost() -> None:
    limiter = InMemoryRateLimiter()
    limit = RateLimit(capacity=2, refill_rate=1.0)
    with pytest.raises(ValueError, match="cost must be positive"):
        await limiter.acquire("k", limit, cost=0)
    with pytest.raises(ValueError, match="cost must be positive"):
        await limiter.acquire("k", limit, cost=-1)


async def test_in_memory_rate_limit_evict_idle_bounds_memory() -> None:
    clock = {"t": 0.0}
    limiter = InMemoryRateLimiter(time_source=lambda: clock["t"])
    limit = RateLimit(capacity=2, refill_rate=1.0)
    for i in range(50):
        assert await limiter.acquire(f"key-{i}", limit)  # 1 token left each
    assert len(limiter._buckets) == 50
    # Nothing has refilled yet, so a sweep now frees nothing.
    assert limiter.evict_idle(limit) == 0
    assert len(limiter._buckets) == 50
    # After a full refill window every bucket is back at capacity and is freed.
    clock["t"] = 10.0
    assert limiter.evict_idle(limit) == 50
    assert len(limiter._buckets) == 0


async def test_in_memory_sweep_interval_triggers_eviction_on_use() -> None:
    clock = {"t": 0.0}
    limiter = InMemoryRateLimiter(time_source=lambda: clock["t"], sweep_interval=5.0)
    limit = RateLimit(capacity=2, refill_rate=1.0)
    # First acquire arms the gate (no sweep on a cold map).
    for i in range(50):
        assert await limiter.acquire(f"key-{i}", limit)
    assert len(limiter._buckets) == 50
    # Within the interval, no opportunistic sweep happens even after refill.
    clock["t"] = 4.0
    assert await limiter.acquire("trigger", limit)
    assert len(limiter._buckets) == 51
    # Once the interval elapses, the next acquire sweeps fully-refilled buckets.
    clock["t"] = 12.0
    assert await limiter.acquire("trigger", limit)
    # Only the active "trigger" bucket (just spent a token) survives the sweep.
    assert list(limiter._buckets) == ["trigger"]


async def test_in_memory_sweep_disabled_by_default() -> None:
    clock = {"t": 0.0}
    limiter = InMemoryRateLimiter(time_source=lambda: clock["t"])
    limit = RateLimit(capacity=2, refill_rate=1.0)
    for i in range(10):
        assert await limiter.acquire(f"key-{i}", limit)
    clock["t"] = 1000.0
    # No sweep_interval: the map is never swept opportunistically.
    assert await limiter.acquire("another", limit)
    assert len(limiter._buckets) == 11


def test_rate_limit_rejects_nonpositive_config() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        RateLimit(capacity=0, refill_rate=1.0)
    with pytest.raises(ValueError, match="refill_rate must be positive"):
        RateLimit(capacity=1.0, refill_rate=0)


class _ScriptedRedis:
    """Stands in for Redis ``eval``: records each call and returns a queued result.

    fakeredis does not implement Lua ``EVAL``, so the token-bucket script cannot
    run against it. This fake instead lets a test assert the exact arguments the
    limiter passes through and control the allow/deny result the script returns.
    """

    def __init__(self, results: list[int]) -> None:
        self._results = results
        self.calls: list[tuple[object, ...]] = []

    async def eval(self, script: str, numkeys: int, *args: object) -> int:
        self.calls.append((script, numkeys, *args))
        return self._results.pop(0)


async def test_redis_limiter_passes_script_args_and_coerces_result() -> None:
    redis = _ScriptedRedis(results=[1, 0])
    limiter = RedisRateLimiter(redis, prefix="optimus:test")
    limit = RateLimit(capacity=4.0, refill_rate=2.0)

    assert await limiter.acquire("guild:9", limit, cost=3.0) is True
    assert await limiter.acquire("guild:9", limit) is False  # returns 0 -> denied

    _script, numkeys, key, capacity, refill, cost, now = redis.calls[0]
    assert numkeys == 1
    assert key == "optimus:test:guild:9"  # prefix applied
    assert (capacity, refill, cost) == (4.0, 2.0, 3.0)
    assert isinstance(now, float)  # wall-clock timestamp passed as ARGV[4]
    # Default cost is 1.0 on the second call.
    assert redis.calls[1][5] == 1.0


async def test_redis_rate_limit_rejects_nonpositive_cost() -> None:
    limiter = RedisRateLimiter(_ScriptedRedis(results=[]))
    limit = RateLimit(capacity=2.0, refill_rate=1.0)
    with pytest.raises(ValueError, match="cost must be positive"):
        await limiter.acquire("k", limit, cost=0)
    with pytest.raises(ValueError, match="cost must be positive"):
        await limiter.acquire("k", limit, cost=-1)


class _LuaBucketRedis:
    """A fake ``eval`` that executes the token-bucket *semantics* in Python.

    fakeredis ships no Lua interpreter, so to exercise refill/burst/atomicity we
    reimplement the script's read-modify-write here against an in-process hash.
    The single ``eval`` coroutine performs the whole bucket update with no await
    between the read and the write, mirroring the server-side atomicity of a real
    Redis ``EVAL`` — that is what lets the concurrency test assert no overspend.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, float]] = {}
        self.calls = 0

    async def eval(self, _script: str, _numkeys: int, *args: object) -> int:
        self.calls += 1
        key = str(args[0])
        capacity, refill, cost, now = (float(a) for a in args[1:5])  # type: ignore[arg-type]
        tokens, ts = self._buckets.get(key, (capacity, now))
        elapsed = max(0.0, now - ts)
        tokens = min(capacity, tokens + elapsed * refill)
        allowed = 0
        if tokens >= cost:
            allowed = 1
            tokens -= cost
        self._buckets[key] = (tokens, now)
        return allowed


async def test_redis_limiter_burst_then_refill(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = _LuaBucketRedis()
    limiter = RedisRateLimiter(redis)
    limit = RateLimit(capacity=2.0, refill_rate=1.0)
    # Pin the wall clock the limiter passes as ARGV[4] so refill is deterministic.
    t = {"now": 1000.0}
    monkeypatch.setattr("optimus.core.ratelimit.time.time", lambda: t["now"])

    assert await limiter.acquire("g", limit) is True  # burst 1/2
    assert await limiter.acquire("g", limit) is True  # burst 2/2
    assert await limiter.acquire("g", limit) is False  # drained
    t["now"] += 1.0  # one token refilled
    assert await limiter.acquire("g", limit) is True
    assert await limiter.acquire("g", limit) is False


async def test_redis_limiter_concurrent_acquire_is_atomic() -> None:
    # capacity 5, fired concurrently 20x: exactly 5 must be allowed, no overspend.
    redis = _LuaBucketRedis()
    limiter = RedisRateLimiter(redis)
    limit = RateLimit(capacity=5.0, refill_rate=0.001)  # negligible refill in-test
    results = await asyncio.gather(*(limiter.acquire("g", limit) for _ in range(20)))
    assert sum(results) == 5
    assert redis.calls == 20


class _ExplodingRedis:
    """``eval`` always raises, simulating a runtime Redis outage."""

    async def eval(self, *_args: object) -> int:
        raise ConnectionError("redis down")


async def test_redis_limiter_falls_back_to_in_memory_on_error() -> None:
    from optimus.core.ratelimit import REDIS_RATELIMIT_FALLBACK

    fallback = InMemoryRateLimiter()
    limiter = RedisRateLimiter(_ExplodingRedis(), fallback=fallback)
    limit = RateLimit(capacity=1.0, refill_rate=0.0001)

    before = REDIS_RATELIMIT_FALLBACK._value.get()
    assert await limiter.acquire("k", limit) is True  # served by fallback
    assert await limiter.acquire("k", limit) is False  # fallback now drained
    assert REDIS_RATELIMIT_FALLBACK._value.get() == before + 2
    # The fallback owns the bucket state, proving the request path was served.
    assert "k" in fallback._buckets


async def test_redis_limiter_reraises_when_no_fallback() -> None:
    limiter = RedisRateLimiter(_ExplodingRedis())
    limit = RateLimit(capacity=1.0, refill_rate=1.0)
    with pytest.raises(ConnectionError):
        await limiter.acquire("k", limit)


async def test_evict_idle_keeps_actively_throttled_buckets() -> None:
    clock = {"t": 0.0}
    limiter = InMemoryRateLimiter(time_source=lambda: clock["t"])
    limit = RateLimit(capacity=3, refill_rate=1.0)
    assert await limiter.acquire("busy", limit)
    assert await limiter.acquire("busy", limit)
    assert await limiter.acquire("busy", limit)  # drained to 0
    clock["t"] = 1.0  # only 1 token back -> still below capacity
    assert limiter.evict_idle(limit) == 0
    assert "busy" in limiter._buckets


# --- idempotency ---------------------------------------------------------------


class _FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def set(self, key: str, value: str, *, nx: bool, ex: int) -> bool | None:
        if nx and key in self._store:
            return None
        self._store[key] = value
        return True

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    async def delete(self, key: str) -> int:
        return 1 if self._store.pop(key, None) is not None else 0


def test_build_key_format() -> None:
    assert build_key(11, 22) == "optimus:idem:11:22"


def test_idempotency_guard_rejects_nonpositive_ttl() -> None:
    with pytest.raises(ValueError, match="ttl_seconds must be >= 1"):
        IdempotencyGuard(_FakeRedis(), ttl_seconds=0)


async def test_idempotency_guard_acquires_once() -> None:
    guard = IdempotencyGuard(_FakeRedis(), ttl_seconds=60)
    key = build_key(1, 2)
    assert await guard.acquire(key) is True
    assert await guard.acquire(key) is False
    assert await guard.seen(key) is True
    await guard.release(key)
    assert await guard.seen(key) is False


class _ExplodingSetRedis:
    """``set`` always raises, simulating a Redis outage during acquire."""

    def __init__(self) -> None:
        self.calls = 0

    async def set(self, *_args: object, **_kwargs: object) -> bool | None:
        self.calls += 1
        raise ConnectionError("redis down")


async def test_idempotency_guard_fails_open_on_redis_error() -> None:
    from optimus.core.idempotency import REDIS_IDEMPOTENCY_FALLBACK

    redis = _ExplodingSetRedis()
    guard = IdempotencyGuard(redis, ttl_seconds=60)
    before = REDIS_IDEMPOTENCY_FALLBACK._value.get()

    # Fails open (permits processing) rather than propagating the error, so a
    # Redis outage degrades dedup instead of nak-storming the consumer. Each
    # fail-open is counted so the degradation is observable.
    assert await guard.acquire(build_key(1, 2)) is True
    assert await guard.acquire(build_key(3, 4)) is True
    assert redis.calls == 2
    assert REDIS_IDEMPOTENCY_FALLBACK._value.get() == before + 2


class _HangingRedis:
    """``set`` blocks forever, simulating a Redis client wedged on reconnect."""

    async def set(self, *_args: object, **_kwargs: object) -> bool | None:
        await asyncio.sleep(3600)
        return True


async def test_idempotency_guard_fails_open_on_timeout() -> None:
    from optimus.core.idempotency import REDIS_IDEMPOTENCY_FALLBACK

    guard = IdempotencyGuard(_HangingRedis(), ttl_seconds=60, op_timeout=0.05)
    before = REDIS_IDEMPOTENCY_FALLBACK._value.get()

    # A hung Redis call must not pin the caller: op_timeout bounds it and the
    # guard fails open instead of stalling the consumer's in-flight slot.
    assert await guard.acquire(build_key(5, 6)) is True
    assert REDIS_IDEMPOTENCY_FALLBACK._value.get() == before + 1


def test_idempotency_guard_rejects_nonpositive_op_timeout() -> None:
    with pytest.raises(ValueError, match="op_timeout must be > 0"):
        IdempotencyGuard(_FakeRedis(), op_timeout=0)
