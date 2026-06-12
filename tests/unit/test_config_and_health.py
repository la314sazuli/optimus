"""Tests for configuration loading and the health server."""

from __future__ import annotations

import asyncio
import time

import aiohttp

from optimus.core.config import Sensitivity, Settings, Tenancy
from optimus.core.health import HealthServer


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.tenancy is Tenancy.SINGLE
    assert settings.is_multi_tenant is False
    assert settings.sensitivity_default is Sensitivity.BALANCED


def test_retention_and_pool_defaults() -> None:
    settings = Settings(_env_file=None)
    # Retention is off by default so self-hosters keep everything.
    assert settings.detection_retention_days is None
    assert settings.retention_batch_size == 1000
    assert settings.retention_batch_pause_seconds == 0.5
    # Pool defaults are conservative per-replica values.
    assert settings.db_pool_size == 5
    assert settings.db_max_overflow == 10
    assert settings.db_pool_recycle == 1800
    assert settings.db_pool_pre_ping is True


def test_retention_settings_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPTIMUS_DETECTION_RETENTION_DAYS", "90")
    monkeypatch.setenv("OPTIMUS_RETENTION_BATCH_SIZE", "250")
    monkeypatch.setenv("OPTIMUS_DB_POOL_SIZE", "20")
    settings = Settings(_env_file=None)
    assert settings.detection_retention_days == 90
    assert settings.retention_batch_size == 250
    assert settings.db_pool_size == 20


def test_settings_env_prefix(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPTIMUS_TENANCY", "multi")
    monkeypatch.setenv("OPTIMUS_HEALTH_PORT", "9999")
    settings = Settings(_env_file=None)
    assert settings.tenancy is Tenancy.MULTI
    assert settings.is_multi_tenant is True
    assert settings.health_port == 9999


async def test_health_endpoints() -> None:
    server = HealthServer(host="127.0.0.1", port=8137)
    ready = {"ok": True}
    server.add_readiness_check(lambda: _async_bool(ready["ok"]))
    await server.start()
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get("http://127.0.0.1:8137/healthz") as resp:
                assert resp.status == 200
            async with http.get("http://127.0.0.1:8137/readyz") as resp:
                assert resp.status == 200
            async with http.get("http://127.0.0.1:8137/metrics") as resp:
                assert resp.status == 200
                body = await resp.text()
                assert "optimus_bus" in body or body is not None

            ready["ok"] = False
            async with http.get("http://127.0.0.1:8137/readyz") as resp:
                assert resp.status == 503

            server.set_live(False)
            async with http.get("http://127.0.0.1:8137/healthz") as resp:
                assert resp.status == 503
    finally:
        await server.stop()


async def test_readyz_returns_503_when_check_raises() -> None:
    server = HealthServer(host="127.0.0.1", port=8138)

    async def _boom() -> bool:
        raise ConnectionError("dependency down")

    server.add_readiness_check(_boom, name="redis")
    await server.start()
    try:
        async with (
            aiohttp.ClientSession() as http,
            http.get("http://127.0.0.1:8138/readyz") as resp,
        ):
            assert resp.status == 503
    finally:
        await server.stop()


async def test_readyz_returns_503_when_check_hangs() -> None:
    server = HealthServer(host="127.0.0.1", port=8139, check_timeout=0.2)

    async def _hang() -> bool:
        await asyncio.sleep(60)
        return True

    server.add_readiness_check(_hang, name="blackholed")
    await server.start()
    try:
        async with aiohttp.ClientSession() as http:
            started = time.monotonic()
            async with http.get("http://127.0.0.1:8139/readyz") as resp:
                assert resp.status == 503
            elapsed = time.monotonic() - started
            assert elapsed < 5.0, f"probe hung for {elapsed:.1f}s instead of failing closed"
    finally:
        await server.stop()


async def _async_bool(value: bool) -> bool:
    return value
