"""Contract tests binding :class:`HikariRestActions` to hikari's real signatures.

The production incident behind this module: the executor called
``ban_member(guild, user, reason)`` positionally on the raw hikari client, whose
``reason`` is keyword-only — every ban raised ``TypeError``, and protocol-shaped
test doubles could never catch it. These tests therefore mock the *real*
``RESTClientImpl`` with ``create_autospec``, which enforces hikari's actual
method signatures: a call that would not bind against the installed hikari
raises here exactly as it would in production.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import create_autospec

import hikari
import pytest
from hikari.impl.rest import RESTClientImpl

from optimus.services.moderation.rest_adapter import HikariRestActions


@pytest.fixture
def rest() -> Any:
    return create_autospec(RESTClientImpl, instance=True)


async def test_ban_member_binds_with_keyword_reason(rest: Any) -> None:
    await HikariRestActions(rest).ban_member(1, 2, "scam")
    rest.ban_user.assert_awaited_once_with(1, 2, delete_message_seconds=0, reason="scam")


async def test_ban_member_forwards_purge_window(rest: Any) -> None:
    await HikariRestActions(rest).ban_member(1, 2, "scam", purge_seconds=86400)
    rest.ban_user.assert_awaited_once_with(1, 2, delete_message_seconds=86400, reason="scam")


async def test_kick_member_binds_with_keyword_reason(rest: Any) -> None:
    await HikariRestActions(rest).kick_member(1, 2, "scam")
    rest.kick_user.assert_awaited_once_with(1, 2, reason="scam")


async def test_unban_member_binds_with_keyword_reason(rest: Any) -> None:
    await HikariRestActions(rest).unban_member(1, 2, "appeal")
    rest.unban_user.assert_awaited_once_with(1, 2, reason="appeal")


async def test_timeout_member_maps_to_edit_member_disabled_until(rest: Any) -> None:
    before = datetime.now(UTC)
    await HikariRestActions(rest).timeout_member(1, 2, 3600)
    rest.edit_member.assert_awaited_once()
    args, kwargs = rest.edit_member.await_args
    assert args == (1, 2)
    until = kwargs["communication_disabled_until"]
    assert (until - before).total_seconds() == pytest.approx(3600, abs=5)


async def test_send_dm_creates_channel_then_message(rest: Any) -> None:
    await HikariRestActions(rest).send_dm(42, "warning")
    rest.create_dm_channel.assert_awaited_once_with(42)
    channel = rest.create_dm_channel.return_value
    rest.create_message.assert_awaited_once_with(channel.id, "warning")


async def test_delete_message_passes_through(rest: Any) -> None:
    await HikariRestActions(rest).delete_message(10, 20)
    rest.delete_message.assert_awaited_once_with(10, 20)


async def test_delete_message_treats_not_found_as_success(rest: Any) -> None:
    rest.delete_message.side_effect = hikari.NotFoundError("http://x", {}, b"", "gone")
    # Must not raise: an already-deleted message means the goal state is reached,
    # and a manual re-run should proceed to the punitive step instead of erroring.
    await HikariRestActions(rest).delete_message(10, 20)


async def test_non_protocol_attributes_delegate_to_hikari(rest: Any) -> None:
    adapter = HikariRestActions(rest)
    # The coordinator's target resolver and report poster share this object.
    assert adapter.fetch_member is rest.fetch_member
    assert adapter.fetch_guild is rest.fetch_guild
    assert adapter.fetch_roles is rest.fetch_roles
    assert adapter.create_message is rest.create_message
