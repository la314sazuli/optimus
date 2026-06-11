"""Streaming, SSRF-hardened image fetcher built on aiohttp.

Connections are made to the IP pinned by :mod:`optimus.ingest.ssrf` (DNS is
resolved once, up front) while the original host is preserved for TLS SNI and
the ``Host`` header. Redirects are followed manually so each hop is re-validated
through the guard. The body is read in bounded chunks and the fetch is aborted
the moment the size cap is exceeded, so an oversized response never lands fully
in memory. Raw bytes are returned to the caller and never written to disk.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult
from yarl import URL

from optimus.core.logging import get_logger
from optimus.ingest.ssrf import (
    ALLOWED_CONTENT_TYPES,
    PinnedTarget,
    SSRFError,
    validate_url,
)

_log = get_logger(__name__)

# Magic-byte signatures for the formats we accept. (offset, signature) pairs;
# WebP additionally checks the "WEBP" tag at offset 8.
_MAGIC: tuple[tuple[int, bytes], ...] = (
    (0, b"\x89PNG\r\n\x1a\n"),  # PNG
    (0, b"\xff\xd8\xff"),  # JPEG
    (0, b"GIF87a"),  # GIF
    (0, b"GIF89a"),  # GIF
    (0, b"BM"),  # BMP
)


class FetchError(Exception):
    """Raised when an image cannot be safely fetched or validated."""


@dataclass(frozen=True, slots=True)
class FetchedImage:
    """A fetched, size- and type-validated image."""

    data: bytes
    content_type: str
    final_url: str


def sniff_content_type(data: bytes) -> str | None:
    """Return a normalized content type from magic bytes, or ``None``."""
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    for offset, sig in _MAGIC:
        if data[offset : offset + len(sig)] == sig:
            if sig.startswith(b"\x89PNG"):
                return "image/png"
            if sig.startswith(b"\xff\xd8"):
                return "image/jpeg"
            if sig.startswith(b"GIF"):
                return "image/gif"
            if sig.startswith(b"BM"):
                return "image/bmp"
    return None


def _build_connector(target: PinnedTarget) -> aiohttp.TCPConnector:
    """Build a connector that pins ``target.host`` to its resolved IP."""
    resolved: list[ResolveResult] = [
        ResolveResult(
            hostname=target.host,
            host=target.ip,
            port=target.port,
            family=target.family,
            proto=0,
            flags=0,
        )
    ]
    return aiohttp.TCPConnector(resolver=_StaticResolver(resolved), ttl_dns_cache=0)


class _StaticResolver(AbstractResolver):
    """An aiohttp resolver that returns a fixed, pre-validated address."""

    def __init__(self, hosts: list[ResolveResult]) -> None:
        self._hosts = hosts

    async def resolve(self, host: str, port: int = 0, family: int = 0) -> list[ResolveResult]:
        return self._hosts

    async def close(self) -> None:
        return None


async def fetch_image(
    url: str,
    *,
    max_bytes: int,
    max_redirects: int = 3,
    total_timeout: float = 15.0,
) -> FetchedImage:
    """Fetch and validate the image at ``url``.

    Raises :class:`FetchError` (or :class:`SSRFError`) on any policy violation:
    blocked address, disallowed scheme, too many redirects, oversize body, or a
    content type that fails either the header allowlist or magic-byte sniff.
    """
    timeout = aiohttp.ClientTimeout(total=total_timeout)
    current = url
    seen = 0
    while True:
        target = validate_url(current)
        connector = _build_connector(target)
        ssl_ctx: ssl.SSLContext | bool = (
            ssl.create_default_context() if target.scheme == "https" else False
        )
        try:
            async with (
                aiohttp.ClientSession(connector=connector, timeout=timeout) as session,
                session.get(
                    current,
                    allow_redirects=False,
                    ssl=ssl_ctx,
                    headers={"Accept": "image/*"},
                ) as resp,
            ):
                if resp.status in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        raise FetchError("redirect without Location header")
                    seen += 1
                    if seen > max_redirects:
                        raise FetchError("too many redirects")
                    current = str(resp.url.join(URL(location)))
                    continue
                if resp.status != 200:
                    raise FetchError(f"unexpected status {resp.status}")
                return await _read_validated(resp, max_bytes=max_bytes)
        except aiohttp.ClientError as exc:
            raise FetchError(f"transport error: {exc}") from exc


async def _read_validated(resp: aiohttp.ClientResponse, *, max_bytes: int) -> FetchedImage:
    """Stream the body under a hard size cap and validate the content type."""
    header_ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if header_ct and header_ct not in ALLOWED_CONTENT_TYPES:
        raise FetchError(f"disallowed content type: {header_ct!r}")

    declared = resp.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise FetchError("content-length exceeds cap")
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.content.iter_chunked(64 * 1024):
        total += len(chunk)
        if total > max_bytes:
            resp.close()  # abort mid-stream; do not buffer the rest
            raise FetchError("body exceeds size cap")
        chunks.append(chunk)
    data = b"".join(chunks)

    sniffed = sniff_content_type(data)
    if sniffed is None:
        raise FetchError("content failed magic-byte validation")
    return FetchedImage(data=data, content_type=sniffed, final_url=str(resp.url))


__all__ = ["FetchError", "FetchedImage", "SSRFError", "fetch_image", "sniff_content_type"]
