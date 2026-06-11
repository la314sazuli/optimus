"""Integration tests for the streaming, SSRF-pinned image fetcher.

A local aiohttp server stands in for the remote host; the SSRF guard is patched
so the loopback test address is treated as a validated, pinned target. This lets
the redirect re-validation, size-cap abort, and content-type/magic-byte checks
run against real HTTP responses.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from optimus.ingest import fetcher
from optimus.ingest.fetcher import FetchError, fetch_image, sniff_content_type
from optimus.ingest.ssrf import PinnedTarget, SSRFError

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
GIF_BYTES = b"GIF89a" + b"\x00" * 32
NOT_IMAGE = b"<html>nope</html>" + b"\x00" * 32


def _pin(url: str) -> PinnedTarget:
    """A pinned target aimed at the loopback test server."""
    return PinnedTarget(
        url=url, scheme="http", host="example.test", port=0, ip="127.0.0.1", family=socket.AF_INET
    )


@pytest.fixture(autouse=True)
def _patch_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat every URL as a validated loopback target (guard tested separately)."""
    monkeypatch.setattr(fetcher, "validate_url", _pin)


async def _serve(app: web.Application) -> AsyncIterator[str]:
    server = TestServer(app)
    await server.start_server()
    try:
        yield f"http://127.0.0.1:{server.port}"
    finally:
        await server.close()


def test_sniff_content_type() -> None:
    assert sniff_content_type(PNG_BYTES) == "image/png"
    assert sniff_content_type(GIF_BYTES) == "image/gif"
    assert sniff_content_type(b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 8) == "image/webp"
    assert sniff_content_type(NOT_IMAGE) is None


async def test_fetch_ok() -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.Response(body=PNG_BYTES, content_type="image/png")

    app = web.Application()
    app.router.add_get("/x.png", handler)
    gen = _serve(app)
    base = await anext(gen)
    try:
        result = await fetch_image(f"{base}/x.png", max_bytes=1_000_000)
        assert result.content_type == "image/png"
        assert result.data == PNG_BYTES
    finally:
        with pytest.raises(StopAsyncIteration):
            await anext(gen)


async def test_fetch_rejects_disallowed_content_type() -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.Response(body=PNG_BYTES, content_type="text/html")

    app = web.Application()
    app.router.add_get("/x", handler)
    gen = _serve(app)
    base = await anext(gen)
    try:
        with pytest.raises(FetchError):
            await fetch_image(f"{base}/x", max_bytes=1_000_000)
    finally:
        with pytest.raises(StopAsyncIteration):
            await anext(gen)


async def test_fetch_rejects_non_image_bytes() -> None:
    async def handler(_req: web.Request) -> web.Response:
        # Header says image, body is not — magic-byte sniff must reject.
        return web.Response(body=NOT_IMAGE, content_type="image/png")

    app = web.Application()
    app.router.add_get("/x", handler)
    gen = _serve(app)
    base = await anext(gen)
    try:
        with pytest.raises(FetchError):
            await fetch_image(f"{base}/x", max_bytes=1_000_000)
    finally:
        with pytest.raises(StopAsyncIteration):
            await anext(gen)


async def test_fetch_size_cap_aborts_midstream() -> None:
    async def handler(req: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "image/png"})
        await resp.prepare(req)
        # Stream more than the cap in chunks; the fetcher must abort.
        for _ in range(50):
            await resp.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8192)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/big.png", handler)
    gen = _serve(app)
    base = await anext(gen)
    try:
        with pytest.raises(FetchError, match="size cap"):
            await fetch_image(f"{base}/big.png", max_bytes=16 * 1024)
    finally:
        with pytest.raises(StopAsyncIteration):
            await anext(gen)


async def test_fetch_content_length_cap() -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.Response(
            body=PNG_BYTES, content_type="image/png", headers={"Content-Length": "999999"}
        )

    app = web.Application()
    app.router.add_get("/x.png", handler)
    gen = _serve(app)
    base = await anext(gen)
    try:
        with pytest.raises(FetchError, match="content-length"):
            await fetch_image(f"{base}/x.png", max_bytes=1024)
    finally:
        with pytest.raises(StopAsyncIteration):
            await anext(gen)


async def test_fetch_follows_and_revalidates_redirects() -> None:
    async def first(_req: web.Request) -> web.Response:
        raise web.HTTPFound(location="/final.png")

    async def final(_req: web.Request) -> web.Response:
        return web.Response(body=PNG_BYTES, content_type="image/png")

    app = web.Application()
    app.router.add_get("/start", first)
    app.router.add_get("/final.png", final)
    gen = _serve(app)
    base = await anext(gen)
    try:
        result = await fetch_image(f"{base}/start", max_bytes=1_000_000)
        assert result.data == PNG_BYTES
    finally:
        with pytest.raises(StopAsyncIteration):
            await anext(gen)


async def test_fetch_redirect_revalidation_blocks_bad_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    # First hop validates (loopback pin); the redirect target is rejected by the
    # guard, proving each hop is re-validated rather than blindly followed.
    calls = {"n": 0}

    def picky(url: str) -> PinnedTarget:
        calls["n"] += 1
        if calls["n"] == 1:
            return _pin(url)
        raise SSRFError("blocked redirect hop")

    monkeypatch.setattr(fetcher, "validate_url", picky)

    async def first(_req: web.Request) -> web.Response:
        raise web.HTTPFound(location="http://10.0.0.1/evil.png")

    app = web.Application()
    app.router.add_get("/start", first)
    gen = _serve(app)
    base = await anext(gen)
    try:
        with pytest.raises(SSRFError):
            await fetch_image(f"{base}/start", max_bytes=1_000_000)
        assert calls["n"] == 2
    finally:
        with pytest.raises(StopAsyncIteration):
            await anext(gen)


async def test_fetch_too_many_redirects() -> None:
    async def loop(req: web.Request) -> web.Response:
        n = int(req.query.get("n", "0"))
        raise web.HTTPFound(location=f"/loop?n={n + 1}")

    app = web.Application()
    app.router.add_get("/loop", loop)
    gen = _serve(app)
    base = await anext(gen)
    try:
        with pytest.raises(FetchError, match="redirect"):
            await fetch_image(f"{base}/loop", max_bytes=1_000_000, max_redirects=2)
    finally:
        with pytest.raises(StopAsyncIteration):
            await anext(gen)
