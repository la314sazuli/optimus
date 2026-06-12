"""Unit tests for gateway extraction and per-guild scan filtering."""

from __future__ import annotations

from optimus.core.guild_config import GuildConfig, GuildConfigCache
from optimus.services.gateway.extract import (
    Attachment,
    IncomingMessage,
    build_events,
    extract_image_urls_from_content,
)

CID = "test-correlation"


def _msg(**kw: object) -> IncomingMessage:
    base: dict[str, object] = {
        "guild_id": 1,
        "channel_id": 2,
        "message_id": 3,
        "author_id": 4,
    }
    base.update(kw)
    return IncomingMessage(**base)  # type: ignore[arg-type]


def test_extract_image_urls_from_content() -> None:
    text = "look https://x.test/a.png and https://x.test/page and http://y.test/b.JPG?z=1"
    urls = extract_image_urls_from_content(text)
    assert urls == ["https://x.test/a.png", "http://y.test/b.JPG?z=1"]


def test_build_events_from_attachments_by_content_type() -> None:
    att = Attachment(id=10, url="https://cdn.test/a", filename="a", content_type="image/png")
    events = build_events(_msg(attachments=(att,)), correlation_id=CID)
    assert len(events) == 1
    assert events[0].attachment_id == 10
    assert events[0].url == "https://cdn.test/a"


def test_build_events_skips_non_image_attachment() -> None:
    att = Attachment(
        id=10, url="https://cdn.test/a.txt", filename="a.txt", content_type="text/plain"
    )
    events = build_events(_msg(attachments=(att,)), correlation_id=CID)
    assert events == []


def test_build_events_dedups_by_url() -> None:
    att = Attachment(id=10, url="https://cdn.test/a.png", filename="a.png")
    msg = _msg(attachments=(att,), embed_image_urls=("https://cdn.test/a.png",))
    events = build_events(msg, correlation_id=CID)
    assert len(events) == 1


def test_build_events_synthetic_id_is_stable_and_positive() -> None:
    msg = _msg(embed_image_urls=("https://cdn.test/e.png",))
    a = build_events(msg, correlation_id=CID)[0]
    b = build_events(msg, correlation_id=CID)[0]
    assert a.attachment_id == b.attachment_id
    assert 0 < a.attachment_id < (1 << 63)


def test_build_events_from_content_urls() -> None:
    msg = _msg(content="scam here https://x.test/free.gif now")
    events = build_events(msg, correlation_id=CID)
    assert [e.url for e in events] == ["https://x.test/free.gif"]


def test_build_events_caps_attachment_count() -> None:
    from optimus.services.gateway.extract import IMAGES_DROPPED

    atts = tuple(
        Attachment(id=100 + i, url=f"https://cdn.test/{i}.png", filename=f"{i}.png")
        for i in range(15)
    )
    before = IMAGES_DROPPED.labels(reason="attachment_cap")._value.get()
    events = build_events(_msg(attachments=atts), correlation_id=CID, max_images=10)
    assert len(events) == 10  # only the first 10 of 15 are published
    # The 5 dropped extras are counted under the attachment_cap reason.
    after = IMAGES_DROPPED.labels(reason="attachment_cap")._value.get()
    assert after - before == 5


def test_build_events_cap_counts_content_urls_too() -> None:
    # Cap spans attachments + embed/content URLs; attachments are kept first.
    att = Attachment(id=10, url="https://cdn.test/a.png", filename="a.png")
    msg = _msg(
        attachments=(att,),
        embed_image_urls=("https://cdn.test/b.png", "https://cdn.test/c.png"),
    )
    events = build_events(msg, correlation_id=CID, max_images=2)
    assert len(events) == 2
    assert events[0].url == "https://cdn.test/a.png"


def test_build_events_no_cap_when_unset() -> None:
    atts = tuple(
        Attachment(id=200 + i, url=f"https://cdn.test/u{i}.png", filename=f"u{i}.png")
        for i in range(20)
    )
    events = build_events(_msg(attachments=atts), correlation_id=CID)
    assert len(events) == 20


# --- scan filtering ---------------------------------------------------------


def test_should_scan_default_human() -> None:
    cfg = GuildConfig.default(1)
    assert cfg.should_scan(
        channel_id=2, uploader_id=4, author_role_ids=frozenset(), is_bot=False, is_webhook=False
    )


def test_should_scan_ignored_channel() -> None:
    cfg = GuildConfig(guild_id=1, ignored_channels=frozenset({2}))
    assert not cfg.should_scan(
        channel_id=2, uploader_id=4, author_role_ids=frozenset(), is_bot=False, is_webhook=False
    )


def test_should_scan_trusted_user() -> None:
    cfg = GuildConfig(guild_id=1, trusted_users=frozenset({4}))
    assert not cfg.should_scan(
        channel_id=2, uploader_id=4, author_role_ids=frozenset(), is_bot=False, is_webhook=False
    )


def test_should_scan_ignored_role() -> None:
    cfg = GuildConfig(guild_id=1, ignored_roles=frozenset({99}))
    assert not cfg.should_scan(
        channel_id=2, uploader_id=4, author_role_ids=frozenset({99}), is_bot=False, is_webhook=False
    )


def test_should_scan_bots_opt_out_by_default() -> None:
    cfg = GuildConfig.default(1)
    assert not cfg.should_scan(
        channel_id=2, uploader_id=4, author_role_ids=frozenset(), is_bot=True, is_webhook=False
    )
    assert not cfg.should_scan(
        channel_id=2, uploader_id=4, author_role_ids=frozenset(), is_bot=False, is_webhook=True
    )


def test_should_scan_bots_opt_in() -> None:
    cfg = GuildConfig(guild_id=1, scan_bots=True)
    assert cfg.should_scan(
        channel_id=2, uploader_id=4, author_role_ids=frozenset(), is_bot=True, is_webhook=False
    )


def test_guild_config_json_roundtrip() -> None:
    cfg = GuildConfig(
        guild_id=7,
        scan_bots=True,
        ignored_channels=frozenset({1, 2}),
        ignored_roles=frozenset({3}),
        trusted_users=frozenset({4, 5}),
    )
    assert GuildConfig.from_json(cfg.to_json()) == cfg


# --- config cache -----------------------------------------------------------


class _FakeRedis:
    """A minimal async string store for the guild config cache."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


async def test_config_cache_loads_then_serves_from_cache() -> None:
    calls = {"n": 0}

    class _Loader:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    def loader() -> _Loader:
        return _Loader()

    async def fake_load(_session: object, guild_id: int) -> GuildConfig:
        calls["n"] += 1
        return GuildConfig(guild_id=guild_id, scan_bots=True)

    import optimus.core.guild_config as gc

    redis = _FakeRedis()
    cache = GuildConfigCache(redis, loader)
    orig = gc.load_from_db
    gc.load_from_db = fake_load  # type: ignore[assignment]
    try:
        a = await cache.get(42)
        b = await cache.get(42)  # served from cache, no second load
    finally:
        gc.load_from_db = orig  # type: ignore[assignment]
    assert a.scan_bots is True
    assert b == a
    assert calls["n"] == 1
    assert redis.store  # cache was populated


async def test_config_cache_invalidate_drops_key() -> None:
    redis = _FakeRedis()
    redis.store["optimus:guildcfg:1"] = GuildConfig(guild_id=1).to_json()
    cache = GuildConfigCache(redis, lambda: None)
    await cache.invalidate(1)
    assert "optimus:guildcfg:1" not in redis.store
