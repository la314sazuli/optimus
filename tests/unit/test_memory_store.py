"""MemoryStore: NX/TTL semantics and the active expiry sweep.

The sweep matters because the keys this store mostly holds (idempotency guards,
action-dedup markers) are written once with a long TTL and never read again, so
lazy on-access expiry never reclaims them — without an active sweep the map grows
without bound for the full TTL horizon and beyond.
"""

from __future__ import annotations

from optimus.app.memory import _SWEEP_EVERY, MemoryStore


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


async def test_set_get_and_nx() -> None:
    store = MemoryStore()
    assert await store.set("k", "v") is True
    assert await store.get("k") == "v"
    # NX loses against a live key, leaving the original value untouched.
    assert await store.set("k", "other", nx=True) is None
    assert await store.get("k") == "v"


async def test_lazy_expiry_on_access() -> None:
    clock = _Clock()
    store = MemoryStore(time_source=clock)
    await store.set("k", "v", ex=10)
    clock.t += 11
    assert await store.get("k") is None
    assert await store.exists("k") == 0


async def test_sweep_reclaims_unread_expired_keys() -> None:
    """Expired keys that are never read again are still reclaimed by the sweep.

    This is the leak guard: each write is a unique key, so nothing re-reads them
    to trigger lazy expiry. After their TTL passes, a sweep (triggered once per
    ``_SWEEP_EVERY`` writes) must drop them so the map tracks the live keyspace.
    """
    clock = _Clock()
    store = MemoryStore(time_source=clock)

    for i in range(_SWEEP_EVERY):
        await store.set(f"idem:{i}", "1", nx=True, ex=10)
    # All still live (TTL not yet passed); none have been read back.
    assert len(store._data) == _SWEEP_EVERY

    # Advance past the TTL, then drive one more sweep-interval of *fresh* writes.
    clock.t += 100
    for i in range(_SWEEP_EVERY):
        await store.set(f"fresh:{i}", "1", nx=True, ex=10)

    # The expired idem:* keys are gone; only the fresh batch remains.
    assert all(key.startswith("fresh:") for key in store._data)
    assert len(store._data) == _SWEEP_EVERY


async def test_keys_without_ttl_are_retained() -> None:
    store = MemoryStore()
    for i in range(_SWEEP_EVERY + 5):
        await store.set(f"persistent:{i}", "1")
    # No TTL -> sweep must not touch them.
    assert len(store._data) == _SWEEP_EVERY + 5


async def test_delete_removes_key() -> None:
    store = MemoryStore()
    await store.set("k", "v")
    assert await store.delete("k") == 1
    assert await store.delete("k") == 0
