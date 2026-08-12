"""Tests for optimus.core.metrics."""

from prometheus_client import REGISTRY

from optimus.core import metrics


def _value(metric_name: str, **labels: str) -> float:
    metric = REGISTRY._names_to_collectors[metric_name]
    return metric.labels(**labels)._value.get()


def test_record_detection_latency():
    metrics.record_detection(123, 0.05)
    # Histograms don't expose a simple counter, just verify no exception.


def test_record_decode_failure():
    metrics.record_decode_failure("timeout")
    assert _value("optimus_decode_failures_total", reason="timeout") >= 1.0


def test_record_db_lock_retry():
    metrics.record_db_lock_retry("interactions")
    assert _value("optimus_db_lock_retries_total", service="interactions") >= 1.0


def test_set_outbox_lag():
    metrics.set_outbox_lag(42)
    assert REGISTRY._names_to_collectors["optimus_outbox_lag"]._value.get() == 42.0


def test_set_active_guilds():
    metrics.set_active_guilds(5)
    assert REGISTRY._names_to_collectors["optimus_active_guilds"]._value.get() == 5.0
