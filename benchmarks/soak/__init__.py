"""Production-shaped soak harness for ``OPTIMUS_MODE=simple``.

Boots the real :class:`~optimus.app.simple.SimpleApp` composition (with the same
stubbed Discord edges the composition test uses) and drives sustained, realistic
traffic for a configurable duration while sampling process health every 30s. The
goal is to surface slow leaks (RSS/fd/task growth, latency drift, unbounded
in-memory maps) that a short unit test never reaches.

See :mod:`benchmarks.soak.driver` for the orchestration and
:mod:`benchmarks.soak.metrics` for the sampler.
"""
