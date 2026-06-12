"""Property-based tests for the core algorithmic logic.

Each test asserts an invariant that must hold for *any* input drawn by
``hypothesis`` rather than a hand-picked example. They cover the rate limiter,
circuit breaker, perceptual-hash ensemble, the safe-mode EWMA estimator, the
moderation policy thresholds, and the review action-row layout.

All time-dependent code is driven through injected fake clocks, so nothing here
depends on wall-clock time and the default hypothesis profile stays CI-friendly.
"""

from __future__ import annotations

import asyncio
import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from optimus.contracts.events import Action, Verdict
from optimus.core.circuit import CircuitBreaker, CircuitOpenError, CircuitState
from optimus.core.config import Sensitivity
from optimus.core.ratelimit import InMemoryRateLimiter, RateLimit
from optimus.hashing.ensemble import PRESETS, compare
from optimus.hashing.perceptual import HASH_BITS
from optimus.services.moderation import review as review_mod
from optimus.services.moderation.policy import Decision, PolicyInput, decide
from optimus.services.moderation.review import REVIEW_BUTTONS, ReviewAction, build_action_rows
from optimus.services.moderation.safemode import Baseline, evaluate, update_baseline

# --- rate limiter --------------------------------------------------------------

_costs = st.lists(st.floats(min_value=0.01, max_value=20.0, allow_nan=False), max_size=40)
_capacities = st.floats(min_value=0.5, max_value=100.0, allow_nan=False)
_rates = st.floats(min_value=0.01, max_value=100.0, allow_nan=False)


@settings(max_examples=150)
@given(capacity=_capacities, rate=_rates, costs=_costs, ticks=_costs)
def test_rate_limiter_tokens_stay_within_bounds(
    capacity: float, rate: float, costs: list[float], ticks: list[float]
) -> None:
    """Across any sequence of acquires and clock advances, tokens stay in [0, capacity]."""
    clock = {"t": 0.0}
    limiter = InMemoryRateLimiter(time_source=lambda: clock["t"])
    limit = RateLimit(capacity=capacity, refill_rate=rate)

    asyncio.run(limiter.acquire("k", limit))  # materialise the bucket
    for cost, dt in zip(costs, ticks, strict=False):
        clock["t"] += dt
        asyncio.run(limiter.acquire("k", limit, cost=cost))
        tokens = limiter._buckets["k"].tokens
        assert -1e-9 <= tokens <= capacity + 1e-9


@settings(max_examples=150)
@given(capacity=_capacities, rate=_rates, cost=st.floats(min_value=0.01, max_value=120.0))
def test_rate_limiter_allow_deny_consistency(capacity: float, rate: float, cost: float) -> None:
    """A grant decrements exactly ``cost`` tokens; a denial leaves them untouched."""
    clock = {"t": 0.0}
    limiter = InMemoryRateLimiter(time_source=lambda: clock["t"])
    limit = RateLimit(capacity=capacity, refill_rate=rate)
    asyncio.run(limiter.acquire("k", limit))  # materialise the bucket (consumes default cost=1.0)
    before = limiter._buckets["k"].tokens

    allowed = asyncio.run(limiter.acquire("k", limit, cost=cost))
    after = limiter._buckets["k"].tokens
    if allowed:
        assert after == before - cost
    else:
        assert after == before  # no clock advance => refill is zero
        assert before < cost


@settings(max_examples=150)
@given(
    capacity=_capacities,
    rate=_rates,
    dt=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
)
def test_rate_limiter_refill_is_monotonic(capacity: float, rate: float, dt: float) -> None:
    """Letting more time pass never reduces the token count."""
    clock = {"t": 0.0}
    limiter = InMemoryRateLimiter(time_source=lambda: clock["t"])
    limit = RateLimit(capacity=capacity, refill_rate=rate)
    asyncio.run(limiter.acquire("k", limit, cost=capacity))  # drain to empty
    low = limiter._buckets["k"].tokens
    clock["t"] += dt
    asyncio.run(limiter.acquire("k", limit, cost=0.001))  # triggers a refill, tiny draw
    high = limiter._buckets["k"].tokens + 0.001
    assert high >= low - 1e-9


@settings(max_examples=100)
@given(capacity=_capacities, rate=_rates, n=st.integers(min_value=1, max_value=30))
def test_evict_idle_never_drops_active_buckets(capacity: float, rate: float, n: int) -> None:
    """A bucket that has not fully refilled is never evicted."""
    clock = {"t": 0.0}
    limiter = InMemoryRateLimiter(time_source=lambda: clock["t"])
    limit = RateLimit(capacity=capacity, refill_rate=rate)
    for i in range(n):
        # Drain each bucket to empty so none are idle yet.
        asyncio.run(limiter.acquire(f"k{i}", limit, cost=capacity))
    # No time has passed, so every bucket is below capacity and must survive.
    freed = limiter.evict_idle(limit)
    assert freed == 0
    assert len(limiter._buckets) == n


# --- circuit breaker -----------------------------------------------------------

_events = st.lists(st.sampled_from(["ok", "fail", "advance"]), max_size=60)


@settings(max_examples=200)
@given(
    failure_threshold=st.integers(min_value=1, max_value=6),
    success_threshold=st.integers(min_value=1, max_value=6),
    events=_events,
)
def test_circuit_breaker_invariants_under_arbitrary_sequences(
    failure_threshold: int, success_threshold: int, events: list[str]
) -> None:
    """The breaker never enters an invalid state and the permit counter is bounded."""
    clock = {"t": 0.0}
    cb = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_time=5.0,
        success_threshold=success_threshold,
        time_source=lambda: clock["t"],
    )
    for ev in events:
        if ev == "ok":
            cb.record_success()
        elif ev == "fail":
            cb.record_failure()
        else:
            clock["t"] += 6.0  # always enough to clear the recovery window

        state = cb.state  # also drives the open -> half-open transition
        assert state in (CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN)
        assert 0 <= cb._trials_in_flight <= success_threshold
        assert cb._failures >= 0
        assert cb._successes >= 0
        # allow() must agree with the state's contract.
        if state is CircuitState.OPEN:
            assert cb.allow() is False
        elif state is CircuitState.CLOSED:
            assert cb.allow() is True


@settings(max_examples=100)
@given(
    success_threshold=st.integers(min_value=1, max_value=5),
    concurrent=st.integers(min_value=1, max_value=8),
)
def test_circuit_breaker_half_open_permits_bounded(success_threshold: int, concurrent: int) -> None:
    """No more than ``success_threshold`` trials run concurrently in half-open."""

    async def scenario() -> int:
        clock = {"t": 0.0}
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_time=5.0,
            success_threshold=success_threshold,
            time_source=lambda: clock["t"],
        )
        cb.record_failure()
        clock["t"] = 5.0  # enter half-open on next state read
        assert cb.state is CircuitState.HALF_OPEN

        gate = asyncio.Event()
        running = {"n": 0}

        async def trial() -> int:
            running["n"] += 1
            await gate.wait()
            return 1

        tasks = [asyncio.create_task(_guarded(cb, trial)) for _ in range(concurrent)]
        await asyncio.sleep(0)
        peak = running["n"]
        gate.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        return peak

    peak = asyncio.run(scenario())
    assert peak <= success_threshold


async def _guarded(cb: CircuitBreaker, fn: object) -> int | None:
    try:
        return await cb.call(fn)  # type: ignore[arg-type]
    except CircuitOpenError:
        return None


# --- ensemble confidence -------------------------------------------------------

_dist = st.integers(min_value=0, max_value=HASH_BITS)
_hashset = st.fixed_dictionaries({"phash": _dist, "dhash": _dist, "whash": _dist, "ahash": _dist})


def _candidate_from_distances(distances: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """Build (candidate, known) hash sets whose per-family Hamming distance matches."""
    known = dict.fromkeys(distances, 0)
    candidate = {name: (1 << d) - 1 if d > 0 else 0 for name, d in distances.items()}
    return candidate, known


@settings(max_examples=200)
@given(distances=_hashset, sensitivity=st.sampled_from(list(Sensitivity)))
def test_ensemble_confidence_in_unit_interval(
    distances: dict[str, int], sensitivity: Sensitivity
) -> None:
    """Confidence and score always stay within [0, 1] for any distance vector."""
    candidate, known = _candidate_from_distances(distances)
    result = compare(candidate, known, sensitivity)
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.score <= 1.0


@settings(max_examples=200)
@given(
    a=_hashset,
    extra=st.integers(min_value=1, max_value=HASH_BITS),
    sensitivity=st.sampled_from(list(Sensitivity)),
)
def test_ensemble_confidence_monotonic_in_distance(
    a: dict[str, int], extra: int, sensitivity: Sensitivity
) -> None:
    """Increasing every per-family distance never increases confidence."""
    closer = a
    farther = {name: min(HASH_BITS, d + extra) for name, d in a.items()}
    assume(farther != closer)
    c_close = compare(*_candidate_from_distances(closer), sensitivity)
    c_far = compare(*_candidate_from_distances(farther), sensitivity)
    assert c_far.confidence <= c_close.confidence + 1e-9
    assert c_far.score >= c_close.score - 1e-9


@settings(max_examples=100)
@given(distances=_hashset, sensitivity=st.sampled_from(list(Sensitivity)))
def test_ensemble_verdict_matches_score_bands(
    distances: dict[str, int], sensitivity: Sensitivity
) -> None:
    """The verdict is exactly the band that the score falls into."""
    candidate, known = _candidate_from_distances(distances)
    result = compare(candidate, known, sensitivity)
    preset = PRESETS[sensitivity]
    if result.score <= preset.match_threshold:
        assert result.verdict is Verdict.SCAM
    elif result.score <= preset.ambiguous_ceiling:
        assert result.verdict is Verdict.AMBIGUOUS
    else:
        assert result.verdict is Verdict.CLEAN


# --- safe-mode EWMA estimator --------------------------------------------------

_obs = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
_alpha = st.floats(min_value=0.01, max_value=1.0, allow_nan=False)


@settings(max_examples=200)
@given(observations=st.lists(_obs, max_size=50), alpha=_alpha)
def test_ewma_variance_never_negative(observations: list[float], alpha: float) -> None:
    """The EWMA variance estimator stays non-negative and finite for any input."""
    baseline = Baseline()
    for obs in observations:
        baseline = update_baseline(baseline, obs, alpha=alpha)
        assert baseline.variance >= 0.0
        assert baseline.stddev >= 0.0
        assert baseline.samples >= 0


@settings(max_examples=200)
@given(
    observations=st.lists(_obs, min_size=1, max_size=50),
    alpha=_alpha,
    sigma=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    min_floor=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
)
def test_safemode_evaluate_is_numerically_stable(
    observations: list[float], alpha: float, sigma: float, min_floor: float
) -> None:
    """evaluate never trips before warmup and always returns a finite threshold."""
    baseline = Baseline()
    for i, obs in enumerate(observations):
        decision = evaluate(baseline, obs, sigma=sigma, alpha=alpha, min_floor=min_floor, warmup=3)
        assert math.isfinite(decision.threshold)
        if i < 3:
            assert decision.is_anomaly is False  # never flag during warmup
        baseline = decision.baseline


@settings(max_examples=150)
@given(
    mean=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False),
    sigma=st.floats(min_value=0.5, max_value=8.0, allow_nan=False),
    min_floor=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
)
def test_safemode_observation_at_or_below_threshold_is_not_anomaly(
    mean: float, sigma: float, min_floor: float
) -> None:
    """An observation that does not exceed the threshold is never flagged."""
    baseline = Baseline(mean=mean, variance=mean, samples=10)
    # Observe exactly the mean (well within the band) -> never anomalous.
    decision = evaluate(baseline, mean, sigma=sigma, alpha=0.3, min_floor=min_floor, warmup=3)
    assert decision.is_anomaly is False


# --- moderation policy ---------------------------------------------------------

_threshold = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@settings(max_examples=300)
@given(
    verdict=st.sampled_from(list(Verdict)),
    confidence=_confidence,
    configured_action=st.sampled_from(list(Action)),
    a=_threshold,
    b=_threshold,
    safe_mode=st.booleans(),
)
def test_policy_decision_consistency(
    verdict: Verdict,
    confidence: float,
    configured_action: Action,
    a: float,
    b: float,
    safe_mode: bool,
) -> None:
    """For valid thresholds, AUTO_ACT implies SCAM at/above the auto-act bar."""
    mod_queue, auto_act = sorted((a, b))  # enforce auto_act >= mod_queue
    inp = PolicyInput(
        verdict=verdict,
        confidence=confidence,
        configured_action=configured_action,
        mod_queue_threshold=mod_queue,
        auto_act_threshold=auto_act,
        safe_mode=safe_mode,
    )
    outcome = decide(inp)
    assert outcome.decision in (Decision.NONE, Decision.MOD_QUEUE, Decision.AUTO_ACT)

    if outcome.decision is Decision.AUTO_ACT:
        # Auto-action only for SCAM, above the auto bar, not in safe mode, with a real action.
        assert verdict is Verdict.SCAM
        assert confidence >= auto_act
        assert not safe_mode
        assert configured_action not in (Action.NONE, Action.REPORT_ONLY)
    if verdict not in (Verdict.SCAM, Verdict.AMBIGUOUS):
        assert outcome.decision is Decision.NONE
    if confidence < mod_queue and verdict in (Verdict.SCAM, Verdict.AMBIGUOUS):
        assert outcome.decision is Decision.NONE


@settings(max_examples=100)
@given(
    verdict=st.sampled_from(list(Verdict)),
    confidence=_confidence,
    configured_action=st.sampled_from(list(Action)),
    a=_threshold,
    b=_threshold,
)
def test_policy_safe_mode_never_auto_acts(
    verdict: Verdict, confidence: float, configured_action: Action, a: float, b: float
) -> None:
    """With safe mode on, the engine never returns AUTO_ACT."""
    mod_queue, auto_act = sorted((a, b))
    inp = PolicyInput(
        verdict=verdict,
        confidence=confidence,
        configured_action=configured_action,
        mod_queue_threshold=mod_queue,
        auto_act_threshold=auto_act,
        safe_mode=True,
    )
    assert decide(inp).decision is not Decision.AUTO_ACT


# --- review action rows --------------------------------------------------------


# Building rows calls into hikari, whose builder timings vary run to run; the
# property is about layout, not speed, so the per-example deadline is disabled.
@settings(max_examples=150, deadline=None)
@given(count=st.integers(min_value=1, max_value=40), detection_id=st.integers(min_value=0))
def test_build_action_rows_preserves_buttons(count: int, detection_id: int) -> None:
    """For any button count: no empty rows, every row <= 5, all buttons preserved in order."""
    pool = list(ReviewAction) * 10
    buttons = tuple(pool[:count])

    mp = pytest.MonkeyPatch()
    mp.setattr(review_mod, "REVIEW_BUTTONS", buttons)
    try:
        rows = build_action_rows(detection_id)
    finally:
        mp.undo()

    sizes = [len(row.components) for row in rows]  # type: ignore[attr-defined]
    assert all(0 < n <= 5 for n in sizes)  # no empty, no over-full row
    assert sum(sizes) == count  # every button present
    # Flatten the emitted custom_ids and confirm order matches the input buttons.
    emitted: list[str] = []
    for row in rows:
        for component in row.components:  # type: ignore[attr-defined]
            emitted.append(component.custom_id)
    expected = [f"om:v1:{action.value}:{detection_id}" for action in buttons]
    assert emitted == expected


def test_review_buttons_default_layout_is_valid() -> None:
    """The shipped default button set lays out into valid rows."""
    rows = build_action_rows(1)
    sizes = [len(row.components) for row in rows]  # type: ignore[attr-defined]
    assert sum(sizes) == len(REVIEW_BUTTONS)
    assert all(0 < n <= 5 for n in sizes)
