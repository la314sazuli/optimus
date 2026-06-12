"""A tiny aiohttp health/metrics server exposing /healthz, /readyz, /metrics."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from aiohttp import web
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, CollectorRegistry, generate_latest

from optimus.core.logging import get_logger

ReadinessCheck = Callable[[], Awaitable[bool]]

_log = get_logger(__name__)


class HealthServer:
    """Serves liveness, readiness, and Prometheus metrics endpoints.

    Liveness (``/healthz``) reflects process health. Readiness (``/readyz``)
    runs registered async checks; any failure yields HTTP 503. Each check is
    bounded by ``check_timeout`` seconds and fails closed on timeout, so a
    black-holed dependency cannot wedge the probe handler indefinitely.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",  # noqa: S104 - intended bind for containerized service
        port: int = 8080,
        registry: CollectorRegistry = REGISTRY,
        check_timeout: float = 3.0,
    ) -> None:
        self._host = host
        self._port = port
        self._registry = registry
        self._check_timeout = check_timeout
        self._readiness_checks: list[tuple[str, ReadinessCheck]] = []
        self._live = True
        self._app = web.Application()
        self._app.add_routes(
            [
                web.get("/healthz", self._handle_healthz),
                web.get("/readyz", self._handle_readyz),
                web.get("/metrics", self._handle_metrics),
            ]
        )
        self._runner: web.AppRunner | None = None

    def add_readiness_check(self, check: ReadinessCheck, *, name: str = "check") -> None:
        """Register an async readiness check returning ``True`` when ready.

        ``name`` labels the dependency in readiness failure logs (e.g. ``redis``).
        """
        self._readiness_checks.append((name, check))

    def set_live(self, live: bool) -> None:
        """Set process liveness (used to fail ``/healthz`` during shutdown)."""
        self._live = live

    async def _handle_healthz(self, _request: web.Request) -> web.Response:
        if self._live:
            return web.json_response({"status": "ok"})
        return web.json_response({"status": "shutting_down"}, status=503)

    async def _handle_readyz(self, _request: web.Request) -> web.Response:
        for name, check in self._readiness_checks:
            try:
                ok = await asyncio.wait_for(check(), timeout=self._check_timeout)
            except TimeoutError:
                # A black-holed dependency (firewall DROP) never raises promptly;
                # fail closed so the probe returns 503 rather than hanging.
                _log.warning("readiness_check_timeout", check=name, timeout=self._check_timeout)
                ok = False
            except Exception:
                _log.exception("readiness_check_errored", check=name)
                ok = False
            if not ok:
                _log.warning("readiness_check_failed", check=name)
                return web.json_response({"status": "not_ready"}, status=503)
        return web.json_response({"status": "ready"})

    async def _handle_metrics(self, _request: web.Request) -> web.Response:
        payload = generate_latest(self._registry)
        return web.Response(body=payload, content_type=CONTENT_TYPE_LATEST.split(";")[0])

    async def start(self) -> None:
        """Start serving in the current event loop."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

    async def stop(self) -> None:
        """Stop serving and release resources."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
