"""Preflight validation for simple-mode startup."""

from __future__ import annotations

import os

import pytest

from optimus.app.startup import StartupError, validate_simple_startup
from optimus.core.config import Settings


def _settings(tmp_path, **overrides: object) -> Settings:  # type: ignore[no-untyped-def]
    db = overrides.pop("simple_database_url", f"sqlite+aiosqlite:///{tmp_path / 'optimus.db'}")
    base: dict[str, object] = {
        "mode": "simple",
        "discord_token": "a-valid-looking-token",
        "simple_database_url": db,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_valid_settings_pass(tmp_path) -> None:  # type: ignore[no-untyped-def]
    validate_simple_startup(_settings(tmp_path))


def test_missing_token_is_friendly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(StartupError, match="OPTIMUS_DISCORD_TOKEN is not set"):
        validate_simple_startup(_settings(tmp_path, discord_token=""))


def test_token_with_whitespace_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(StartupError, match="whitespace"):
        validate_simple_startup(_settings(tmp_path, discord_token=" tok "))


def test_quoted_token_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(StartupError, match="quoted"):
        validate_simple_startup(_settings(tmp_path, discord_token='"tok"'))


def test_unwritable_sqlite_dir_is_friendly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    readonly = tmp_path / "ro"
    readonly.mkdir()
    os.chmod(readonly, 0o500)
    try:
        url = f"sqlite+aiosqlite:///{readonly / 'optimus.db'}"
        with pytest.raises(StartupError, match=r"Cannot (write|create)"):
            validate_simple_startup(_settings(tmp_path, simple_database_url=url))
    finally:
        os.chmod(readonly, 0o700)


def test_creates_missing_parent_dir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    url = f"sqlite+aiosqlite:///{tmp_path / 'nested' / 'deep' / 'optimus.db'}"
    validate_simple_startup(_settings(tmp_path, simple_database_url=url))
    assert (tmp_path / "nested" / "deep").is_dir()


def test_in_memory_sqlite_skips_file_check(tmp_path) -> None:  # type: ignore[no-untyped-def]
    url = "sqlite+aiosqlite:///:memory:"
    validate_simple_startup(_settings(tmp_path, simple_database_url=url))


def test_uncreatable_parent_dir_is_friendly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # mkdir of a nested path under a read-only directory fails -> the create-dir branch.
    readonly = tmp_path / "ro"
    readonly.mkdir()
    os.chmod(readonly, 0o500)
    try:
        url = f"sqlite+aiosqlite:///{readonly / 'sub' / 'optimus.db'}"
        with pytest.raises(StartupError, match="Cannot create the directory"):
            validate_simple_startup(_settings(tmp_path, simple_database_url=url))
    finally:
        os.chmod(readonly, 0o700)


def test_non_sqlite_url_skips_file_check(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # In distributed mode the effective URL is Postgres, not SQLite: nothing to check.
    validate_simple_startup(
        _settings(
            tmp_path,
            mode="distributed",
            database_url="postgresql+asyncpg://u:p@localhost:5432/optimus",
        )
    )
