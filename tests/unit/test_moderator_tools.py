"""Tests for moderator tools: /scamhash explain and /scamhash undo."""

from unittest.mock import AsyncMock, MagicMock

from optimus.services.interactions.handlers import InteractionContext, handle_command


def _ctx(command="scamhash", sub="explain", options=None, guild_id=100, user_id=200):
    return InteractionContext(
        command=command,
        subcommand=sub,
        options=options or {"detection_id": 42},
        guild_id=guild_id,
        user_id=user_id,
        member_permissions=0x20,
    )


def _deps():
    deps = MagicMock()
    deps.detection_detail = AsyncMock(
        return_value={
            "detection_id": 42,
            "verdict": "scam",
            "distance": 5,
            "action": "delete",
            "created_at": "2026-08-12T00:00:00",
        }
    )
    deps.last_detection = AsyncMock(
        return_value={
            "detection_id": 42,
            "verdict": "scam",
            "distance": 5,
            "action": "delete",
            "created_at": "2026-08-12T00:00:00",
        }
    )
    deps.reverse_detection_action = AsyncMock(return_value=None)
    deps.audit = AsyncMock(return_value=None)
    return deps


async def test_explain_found():
    ctx = _ctx()
    deps = _deps()
    resp = await handle_command(ctx, deps)
    assert resp.i18n_key == "command.explain_result"
    deps.detection_detail.assert_awaited_once_with(100, 42)


async def test_explain_not_found():
    ctx = _ctx()
    deps = _deps()
    deps.detection_detail.return_value = None
    resp = await handle_command(ctx, deps)
    assert resp.i18n_key == "command.explain_not_found"


async def test_undo_with_detection():
    ctx = _ctx(sub="undo")
    deps = _deps()
    resp = await handle_command(ctx, deps)
    assert resp.i18n_key == "command.undo_done"
    deps.reverse_detection_action.assert_awaited_once_with(100, 42)
    deps.audit.assert_awaited_once()


async def test_undo_nothing():
    ctx = _ctx(sub="undo")
    deps = _deps()
    deps.last_detection.return_value = None
    resp = await handle_command(ctx, deps)
    assert resp.i18n_key == "command.undo_nothing"
    deps.reverse_detection_action.assert_not_awaited()
