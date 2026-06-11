"""SSRF guard for outbound image fetches.

Untrusted URLs (from Discord messages) are fetched only after passing this
guard. The guard resolves DNS itself and pins the resulting IP for the actual
connection, closing the classic DNS-rebinding hole where a name resolves to a
public address during validation and a private address at connect time.

Blocked destinations (IPv4 and IPv6): loopback, private, link-local, CGNAT,
multicast, reserved, unspecified, and the cloud metadata endpoints. Non-Discord
hosts must use HTTPS. Each redirect hop is re-validated against these rules.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

# Discord-controlled CDN/media hosts that are permitted over plain criteria.
# Suffix match (host == suffix or endswith "." + suffix).
DISCORD_HOST_SUFFIXES: tuple[str, ...] = (
    "discord.com",
    "discordapp.com",
    "discordapp.net",
    "cdn.discordapp.com",
    "media.discordapp.net",
)

# Content types we accept (paired with magic-byte sniffing downstream).
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/bmp",
    }
)

# Explicit metadata addresses that must always be blocked even if a future
# library change reclassifies them.
_METADATA_ADDRS: frozenset[str] = frozenset(
    {
        "169.254.169.254",
        "fd00:ec2::254",
    }
)


class SSRFError(ValueError):
    """Raised when a URL or resolved address fails SSRF validation."""


@dataclass(frozen=True, slots=True)
class PinnedTarget:
    """A validated fetch target with its resolved IP pinned for connection."""

    url: str
    scheme: str
    host: str
    port: int
    ip: str
    family: int

    @property
    def is_ipv6(self) -> bool:
        """Whether the pinned address is IPv6."""
        return self.family == socket.AF_INET6


def is_discord_host(host: str) -> bool:
    """Whether ``host`` is a Discord-controlled host (suffix match)."""
    h = host.lower().rstrip(".")
    return any(h == s or h.endswith("." + s) for s in DISCORD_HOST_SUFFIXES)


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an address falls in any range we refuse to connect to."""
    if str(ip) in _METADATA_ADDRS:
        return True
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    if isinstance(ip, ipaddress.IPv4Address):
        # 100.64.0.0/10 — carrier-grade NAT (RFC 6598). Not flagged is_private.
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return True
    else:
        # Unwrap IPv4-mapped/compat IPv6 and re-check against IPv4 rules.
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None and _ip_is_blocked(mapped):
            return True
        if ip in ipaddress.ip_network("fc00::/7"):  # unique local (ULA)
            return True
        if ip in ipaddress.ip_network("fe80::/10"):  # link-local
            return True
    return False


def validate_ip(ip_str: str) -> None:
    """Raise :class:`SSRFError` if ``ip_str`` is a forbidden destination."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError as exc:
        raise SSRFError(f"invalid IP address: {ip_str!r}") from exc
    if _ip_is_blocked(ip):
        raise SSRFError(f"blocked address range: {ip_str}")


def _resolve(host: str, port: int) -> tuple[str, int]:
    """Resolve ``host`` to a single IP, preferring the first usable address.

    Returns ``(ip, family)``. Every returned candidate is validated; if any
    resolved address is blocked the whole fetch is refused (fail closed).
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFError(f"DNS resolution failed for {host!r}") from exc
    if not infos:
        raise SSRFError(f"no addresses for {host!r}")
    # Validate ALL resolved addresses — a host that resolves to any blocked
    # address is rejected outright rather than racing a "good" one.
    for _family, _type, _proto, _canon, sockaddr in infos:
        validate_ip(str(sockaddr[0]))
    family, _type, _proto, _canon, sockaddr = infos[0]
    return str(sockaddr[0]), int(family)


def validate_url(url: str) -> PinnedTarget:
    """Validate ``url`` and return a :class:`PinnedTarget` with a resolved IP.

    Enforces scheme rules (HTTPS-only for non-Discord hosts), resolves DNS, and
    validates every resolved address. The returned target's ``ip`` must be used
    for the actual TCP connection (with ``host`` preserved for SNI/Host header).
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise SSRFError(f"unsupported scheme: {scheme!r}")
    host = parts.hostname
    if not host:
        raise SSRFError("URL has no host")

    discord = is_discord_host(host)
    if not discord and scheme != "https":
        raise SSRFError("non-Discord hosts must use HTTPS")

    port = parts.port or (443 if scheme == "https" else 80)

    # A literal IP in the URL bypasses DNS; validate it directly.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        validate_ip(str(literal))
        family = int(socket.AF_INET6 if literal.version == 6 else socket.AF_INET)
        return PinnedTarget(
            url=url, scheme=scheme, host=host, port=port, ip=str(literal), family=family
        )

    ip, family = _resolve(host, port)
    return PinnedTarget(url=url, scheme=scheme, host=host, port=port, ip=ip, family=family)
