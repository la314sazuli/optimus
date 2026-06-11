"""Unit tests for the SSRF guard."""

from __future__ import annotations

import ipaddress
import socket
from unittest import mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from optimus.ingest import ssrf
from optimus.ingest.ssrf import SSRFError, is_discord_host, validate_ip, validate_url

BLOCKED_IPS = [
    "127.0.0.1",  # loopback
    "10.0.0.5",  # private
    "192.168.1.1",  # private
    "172.16.0.1",  # private
    "169.254.169.254",  # link-local + cloud metadata
    "169.254.1.1",  # link-local
    "100.64.0.1",  # CGNAT (RFC 6598)
    "224.0.0.1",  # multicast
    "0.0.0.0",  # unspecified
    "::1",  # IPv6 loopback
    "fe80::1",  # IPv6 link-local
    "fc00::1",  # IPv6 unique-local
    "fd00::1",  # IPv6 unique-local
    "fd00:ec2::254",  # IPv6 cloud metadata
    "::ffff:127.0.0.1",  # IPv4-mapped loopback
    "::ffff:10.0.0.1",  # IPv4-mapped private
    "::",  # IPv6 unspecified
]

ALLOWED_IPS = [
    "8.8.8.8",
    "1.1.1.1",
    "93.184.216.34",
    "2606:4700:4700::1111",
]


@pytest.mark.parametrize("ip", BLOCKED_IPS)
def test_validate_ip_blocks_dangerous_ranges(ip: str) -> None:
    with pytest.raises(SSRFError):
        validate_ip(ip)


@pytest.mark.parametrize("ip", ALLOWED_IPS)
def test_validate_ip_allows_public(ip: str) -> None:
    validate_ip(ip)  # must not raise


def test_validate_ip_rejects_garbage() -> None:
    with pytest.raises(SSRFError):
        validate_ip("not-an-ip")


@pytest.mark.parametrize(
    "host",
    ["discord.com", "cdn.discordapp.com", "media.discordapp.net", "DISCORD.COM", "x.discord.com"],
)
def test_is_discord_host_true(host: str) -> None:
    assert is_discord_host(host)


@pytest.mark.parametrize("host", ["evil.com", "discord.com.evil.com", "notdiscord.com"])
def test_is_discord_host_false(host: str) -> None:
    assert not is_discord_host(host)


def test_validate_url_rejects_non_https_for_non_discord() -> None:
    with pytest.raises(SSRFError):
        validate_url("http://example.com/x.png")


def test_validate_url_rejects_unsupported_scheme() -> None:
    with pytest.raises(SSRFError):
        validate_url("ftp://example.com/x.png")


def test_validate_url_rejects_no_host() -> None:
    with pytest.raises(SSRFError):
        validate_url("https:///x.png")


def test_validate_url_literal_blocked_ip() -> None:
    with pytest.raises(SSRFError):
        validate_url("http://127.0.0.1/x.png")


def test_validate_url_literal_public_ip_pins() -> None:
    target = validate_url("https://8.8.8.8/x.png")
    assert target.ip == "8.8.8.8"
    assert target.family == socket.AF_INET
    assert not target.is_ipv6


def test_validate_url_pins_resolved_ip() -> None:
    fake = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))]
    with mock.patch.object(ssrf.socket, "getaddrinfo", return_value=fake):
        target = validate_url("https://example.com/x.png")
    assert target.ip == "93.184.216.34"
    assert target.host == "example.com"
    assert target.port == 443


def test_validate_url_rejects_when_any_resolved_ip_is_blocked() -> None:
    # Rebinding-style: one public, one private — fail closed.
    fake = [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.1", 443)),
    ]
    with (
        mock.patch.object(ssrf.socket, "getaddrinfo", return_value=fake),
        pytest.raises(SSRFError),
    ):
        validate_url("https://example.com/x.png")


def test_validate_url_dns_failure() -> None:
    with (
        mock.patch.object(ssrf.socket, "getaddrinfo", side_effect=socket.gaierror("boom")),
        pytest.raises(SSRFError),
    ):
        validate_url("https://does-not-resolve.example/x.png")


@given(st.integers(min_value=0, max_value=(1 << 32) - 1))
def test_property_private_ipv4_always_blocked(value: int) -> None:
    ip = ipaddress.IPv4Address(value)
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
        with pytest.raises(SSRFError):
            validate_ip(str(ip))


@given(st.integers(min_value=0, max_value=(1 << 16) - 1))
def test_property_loopback_ipv6_blocked(_value: int) -> None:
    # The ::1 loopback and fe80:: link-local prefixes must always be refused.
    with pytest.raises(SSRFError):
        validate_ip("::1")
    with pytest.raises(SSRFError):
        validate_ip(f"fe80::{_value:x}")
