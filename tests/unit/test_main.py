"""The ``python -m optimus`` entrypoint: mode dispatch and friendly errors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import optimus.__main__ as entry
from optimus.app.startup import StartupError


def test_distributed_mode_prints_help_and_exits_2(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(entry, "get_settings", lambda: SimpleNamespace(is_simple_mode=False))
    with pytest.raises(SystemExit) as excinfo:
        entry.main()
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "python -m optimus.services.gateway" in err


def test_startup_error_is_friendly(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(entry, "get_settings", lambda: SimpleNamespace(is_simple_mode=True))

    async def _boom() -> None:
        raise StartupError("OPTIMUS_DISCORD_TOKEN is not set")

    monkeypatch.setattr("optimus.app.simple.run_simple", _boom)
    with pytest.raises(SystemExit) as excinfo:
        entry.main()
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "Optimus could not start: OPTIMUS_DISCORD_TOKEN is not set" in err
    assert "Traceback" not in err


def test_keyboard_interrupt_exits_0(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(entry, "get_settings", lambda: SimpleNamespace(is_simple_mode=True))

    async def _interrupt() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("optimus.app.simple.run_simple", _interrupt)
    with pytest.raises(SystemExit) as excinfo:
        entry.main()
    assert excinfo.value.code == 0


def test_simple_mode_runs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(entry, "get_settings", lambda: SimpleNamespace(is_simple_mode=True))
    called = {"ran": False}

    async def _ok() -> None:
        called["ran"] = True

    monkeypatch.setattr("optimus.app.simple.run_simple", _ok)
    entry.main()
    assert called["ran"] is True
