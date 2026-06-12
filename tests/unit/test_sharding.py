"""Tests for gateway sharding: settings validation, start-kwarg wiring, readiness."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from optimus.core.config import Settings, _parse_shard_ids
from optimus.core.readiness import shards_check
from optimus.services.gateway.bot import shard_start_kwargs


def _settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


# --- shard-id spec parsing --------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", (0,)),
        ("0,1,2", (0, 1, 2)),
        ("0-3", (0, 1, 2, 3)),
        ("0-1,4,6-7", (0, 1, 4, 6, 7)),
        (" 2 , 0 , 1 ", (0, 1, 2)),
        ("1,1,2", (1, 2)),
        ("5-5", (5,)),
    ],
)
def test_parse_shard_ids(raw: str, expected: tuple[int, ...]) -> None:
    assert _parse_shard_ids(raw) == expected


def test_parse_shard_ids_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="inverted"):
        _parse_shard_ids("3-1")


def test_parse_shard_ids_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="invalid literal"):
        _parse_shard_ids("abc")


# --- settings validation ----------------------------------------------------


def test_sharding_unset_by_default() -> None:
    s = _settings()
    assert s.shard_count is None
    assert s.shard_ids is None
    assert shard_start_kwargs(s) == {}


def test_shard_count_only_is_valid() -> None:
    s = _settings(shard_count=4)
    assert s.shard_count == 4
    assert s.shard_ids is None
    assert shard_start_kwargs(s) == {"shard_count": 4}


def test_shard_ids_string_is_parsed_and_sorted() -> None:
    s = _settings(shard_count=4, shard_ids="2,0-1")
    assert s.shard_ids == (0, 1, 2)
    assert shard_start_kwargs(s) == {"shard_count": 4, "shard_ids": [0, 1, 2]}


def test_shard_ids_blank_string_is_none() -> None:
    assert _settings(shard_ids="").shard_ids is None
    assert _settings(shard_ids="   ").shard_ids is None


def test_shard_count_blank_string_is_none() -> None:
    assert _settings(shard_count="").shard_count is None  # type: ignore[arg-type]


def test_shard_ids_without_count_rejected() -> None:
    with pytest.raises(ValidationError, match="shard_count is required"):
        _settings(shard_ids="0,1")


def test_shard_id_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError, match="must be < shard_count"):
        _settings(shard_count=2, shard_ids="0,2")


def test_shard_count_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(shard_count=0)


def test_env_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIMUS_SHARD_COUNT", "8")
    monkeypatch.setenv("OPTIMUS_SHARD_IDS", "0-3")
    s = Settings(_env_file=None)
    assert s.shard_count == 8
    assert s.shard_ids == (0, 1, 2, 3)
    assert shard_start_kwargs(s) == {"shard_count": 8, "shard_ids": [0, 1, 2, 3]}


# --- readiness --------------------------------------------------------------


class _FakeShard:
    def __init__(self, *, alive: bool, connected: bool) -> None:
        self.is_alive = alive
        self.is_connected = connected


class _FakeBot:
    def __init__(self, shards: dict[int, _FakeShard]) -> None:
        self.shards = shards


async def test_shards_check_ready_when_all_connected() -> None:
    bot = _FakeBot(
        {0: _FakeShard(alive=True, connected=True), 1: _FakeShard(alive=True, connected=True)}
    )
    assert await shards_check(bot)() is True


async def test_shards_check_not_ready_when_one_disconnected() -> None:
    bot = _FakeBot(
        {0: _FakeShard(alive=True, connected=True), 1: _FakeShard(alive=True, connected=False)}
    )
    assert await shards_check(bot)() is False


async def test_shards_check_not_ready_when_not_alive() -> None:
    bot = _FakeBot({0: _FakeShard(alive=False, connected=True)})
    assert await shards_check(bot)() is False


async def test_shards_check_not_ready_when_no_shards() -> None:
    assert await shards_check(_FakeBot({}))() is False


async def test_shards_check_fails_closed_on_bad_object() -> None:
    assert await shards_check(object())() is False


async def test_shards_check_fails_closed_when_access_raises() -> None:
    class _Boom:
        @property
        def shards(self) -> dict[int, _FakeShard]:
            raise RuntimeError("gateway not started")

    assert await shards_check(_Boom())() is False
