"""Unit tests for the evidence store TTL/key logic and storage flow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from optimus.evidence.store import (
    MAX_TTL_SECONDS,
    EvidenceStore,
    clamp_ttl,
    expiry_at,
    object_key,
)


class _FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.sse_used: list[str] = []

    async def put(self, key: str, data: bytes, *, content_type: str, sse: str) -> None:
        self.objects[key] = data
        self.sse_used.append(sse)

    async def presign_get(self, key: str, *, expires_in: int) -> str:
        return f"https://store.local/{key}?exp={expires_in}"

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.objects.pop(key, None)


def test_object_key_scheme() -> None:
    assert object_key(123, 456) == "evidence/123/456"


def test_clamp_ttl_caps_at_max() -> None:
    assert clamp_ttl(10) == 10
    assert clamp_ttl(10**9) == MAX_TTL_SECONDS


def test_clamp_ttl_rejects_zero() -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        clamp_ttl(0)


def test_expiry_at_uses_clamped_ttl() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert expiry_at(3600, now=now) == now + timedelta(hours=1)
    assert expiry_at(10**9, now=now) == now + timedelta(seconds=MAX_TTL_SECONDS)


async def test_store_writes_encrypted_and_presigns() -> None:
    backend = _FakeStore()
    store = EvidenceStore(backend, sse="AES256", default_ttl=3600, presign_seconds=120)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    result = await store.store(guild_id=1, detection_id=2, data=b"img", now=now)
    assert result.object_key == "evidence/1/2"
    assert backend.objects["evidence/1/2"] == b"img"
    assert backend.sse_used == ["AES256"]
    assert result.expires_at == now + timedelta(hours=1)
    assert "exp=120" in result.presigned_url


async def test_store_clamps_excessive_ttl() -> None:
    backend = _FakeStore()
    store = EvidenceStore(backend, default_ttl=3600, max_ttl=MAX_TTL_SECONDS)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    result = await store.store(guild_id=1, detection_id=2, data=b"x", ttl_seconds=10**9, now=now)
    assert result.expires_at == now + timedelta(seconds=MAX_TTL_SECONDS)


async def test_delete_removes_object() -> None:
    backend = _FakeStore()
    store = EvidenceStore(backend)
    await store.store(guild_id=1, detection_id=2, data=b"x")
    await store.delete("evidence/1/2")
    assert "evidence/1/2" in backend.deleted


async def test_presigned_url_uses_configured_expiry() -> None:
    backend = _FakeStore()
    store = EvidenceStore(backend, presign_seconds=45)
    url = await store.presigned_url("evidence/9/9")
    assert url == "https://store.local/evidence/9/9?exp=45"


def test_s3_object_store_normalizes_empty_endpoint() -> None:
    from optimus.evidence.store import S3ObjectStore

    # An empty endpoint_url collapses to None (use AWS default), region is kept.
    store = S3ObjectStore("mybucket", endpoint_url="", region="eu-west-1")
    assert store._endpoint_url is None
    assert store._region == "eu-west-1"
    assert store._bucket == "mybucket"

    explicit = S3ObjectStore("b", endpoint_url="https://minio.local")
    assert explicit._endpoint_url == "https://minio.local"
