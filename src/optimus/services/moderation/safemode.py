"""Anomaly-driven safe mode.

A guild's per-hour detection rate is tracked as an exponentially-weighted moving
average (EWMA) of both the mean and the variance. When a fresh observation lands
more than ``sigma`` standard deviations above the baseline mean -- and the
baseline itself is above a minimum floor (so a quiet guild's first busy minute
doesn't trip it) -- the guild is flipped into safe mode (report-only).

The math lives in :func:`evaluate` as a pure function over an immutable
:class:`Baseline`; :class:`SafeModeTracker` is the thin Redis-backed wrapper that
persists the baseline between hour buckets.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Baseline:
    """The rolling EWMA mean/variance of a guild's per-bucket detection rate."""

    mean: float = 0.0
    variance: float = 0.0
    #: Number of buckets folded in so far (used to bootstrap).
    samples: int = 0

    @property
    def stddev(self) -> float:
        """The baseline standard deviation."""
        return math.sqrt(max(0.0, self.variance))

    def to_json(self) -> str:
        """Serialize for Redis storage."""
        return json.dumps(
            {"mean": self.mean, "variance": self.variance, "samples": self.samples},
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> Baseline:
        """Deserialize a stored baseline."""
        data = json.loads(raw)
        return cls(
            mean=float(data["mean"]),
            variance=float(data["variance"]),
            samples=int(data["samples"]),
        )


@dataclass(frozen=True, slots=True)
class SafeModeDecision:
    """The result of folding one observation into a baseline."""

    baseline: Baseline
    is_anomaly: bool
    threshold: float


def update_baseline(baseline: Baseline, observation: float, *, alpha: float) -> Baseline:
    """Fold ``observation`` into the EWMA mean and variance."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    delta = observation - baseline.mean
    new_mean = baseline.mean + alpha * delta
    # EWMA of variance (West, 1979 style incremental estimator).
    new_var = (1.0 - alpha) * (baseline.variance + alpha * delta * delta)
    return Baseline(mean=new_mean, variance=new_var, samples=baseline.samples + 1)


def evaluate(
    baseline: Baseline,
    observation: float,
    *,
    sigma: float,
    alpha: float,
    min_floor: float,
    warmup: int = 3,
) -> SafeModeDecision:
    """Decide whether ``observation`` is anomalous, returning the new baseline.

    The observation is judged against the *prior* baseline, then folded in. No
    anomaly is ever flagged before ``warmup`` buckets or while the baseline mean
    sits below ``min_floor`` -- both guard against small-sample false positives.

    The effective standard deviation is floored at ``sqrt(mean)`` (a Poisson
    assumption for count data); without it a perfectly steady stream collapses
    the EWMA variance toward zero and any small bump would trip the detector.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    effective_std = max(baseline.stddev, math.sqrt(max(0.0, baseline.mean)))
    threshold = baseline.mean + sigma * effective_std
    is_anomaly = (
        baseline.samples >= warmup and baseline.mean >= min_floor and observation > threshold
    )
    return SafeModeDecision(
        baseline=update_baseline(baseline, observation, alpha=alpha),
        is_anomaly=is_anomaly,
        threshold=threshold,
    )


class SafeModeTracker:
    """Persists per-guild baselines in Redis and evaluates new observations."""

    def __init__(
        self,
        redis: object,
        *,
        sigma: float = 4.0,
        alpha: float = 0.3,
        min_floor: float = 5.0,
        ttl_seconds: int = 7 * 24 * 3600,
        prefix: str = "optimus:safemode",
    ) -> None:
        self._redis = redis
        self._sigma = sigma
        self._alpha = alpha
        self._min_floor = min_floor
        self._ttl = ttl_seconds
        self._prefix = prefix

    def _key(self, guild_id: int) -> str:
        return f"{self._prefix}:{guild_id}"

    async def observe(self, guild_id: int, observation: float) -> SafeModeDecision:
        """Record a bucket's detection count and report whether it is anomalous."""
        key = self._key(guild_id)
        raw = await self._redis.get(key)  # type: ignore[attr-defined]
        baseline = Baseline.from_json(raw) if raw is not None else Baseline()
        decision = evaluate(
            baseline,
            observation,
            sigma=self._sigma,
            alpha=self._alpha,
            min_floor=self._min_floor,
        )
        await self._redis.set(  # type: ignore[attr-defined]
            key, decision.baseline.to_json(), ex=self._ttl
        )
        return decision
