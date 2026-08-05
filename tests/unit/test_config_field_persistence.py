"""Regression coverage for DbDeps.set_config_field actually persisting to the
Guild row it claims to update, for every field /config set accepts.

This exercises the real InteractionService.DbDeps (via a real aiosqlite
session), not FakeDeps -- FakeDeps.set_config_field just appends to a list
and never touches an ORM object, so it could never have caught the bug this
guards against: "review_channel" (the /config set field name) does not match
Guild.review_channel_id (the actual mapped column), and a naive
setattr(guild, field, value) silently sets an untracked plain attribute
instead of raising or writing anything, so the value was accepted by the
command but never actually saved.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from optimus.core.config import get_settings
from optimus.core.ratelimit import RateLimit
from optimus.db.repositories import GuildRepository
from optimus.services.interactions.service import DbDeps

# Arbitrary synthetic IDs -- this suite runs against an isolated in-memory
# aiosqlite database, so these are not tied to any real Discord guild/channel.
GUILD_ID = 111111111111111111
TEST_CHANNEL_ID = 222222222222222222


class _NoopRateLimiter:
    async def acquire(self, key: str, limit: RateLimit, cost: float = 1.0) -> bool:
        return True


def _make_deps(session: AsyncSession) -> DbDeps:
    return DbDeps(session, _NoopRateLimiter(), get_settings())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "expected_column", "expected_value"),
    [
        ("sensitivity", "strict", "sensitivity", "strict"),
        ("action_policy", "delete_ban", "action_policy", "delete_ban"),
        ("mod_queue_threshold", 0.5, "mod_queue_threshold", 0.5),
        ("retention_days", 14, "retention_days", 14),
        ("locale", "sr", "locale", "sr"),
        ("safe_mode", True, "safe_mode", True),
        ("optin_global_db", True, "optin_global_db", True),
        ("optin_scan_bots", True, "optin_scan_bots", True),
        ("optin_evidence_storage", True, "optin_evidence_storage", True),
        # The regression case: command field name "review_channel" differs
        # from the mapped column "review_channel_id".
        ("review_channel", TEST_CHANNEL_ID, "review_channel_id", TEST_CHANNEL_ID),
    ],
)
async def test_set_config_field_persists_to_the_correct_column(
    session: AsyncSession,
    field: str,
    value: Any,
    expected_column: str,
    expected_value: Any,
) -> None:
    deps = _make_deps(session)
    await deps.set_config_field(GUILD_ID, field, value)

    # Fetch through a fresh repository call (not just trusting the same
    # in-memory object) to prove the write actually reached the DB row.
    guild = await GuildRepository(session).get(GUILD_ID)
    assert guild is not None
    assert getattr(guild, expected_column) == expected_value


async def test_set_review_channel_then_get_config_round_trips(session: AsyncSession) -> None:
    """End-to-end: /config set field:review_channel followed by /config view's
    get_config() must see the same value that was just written -- this is
    exactly the sequence the user ran live when review_channel silently
    failed to persist.
    """
    deps = _make_deps(session)
    await deps.set_config_field(GUILD_ID, "review_channel", TEST_CHANNEL_ID)

    config = await deps.get_config(GUILD_ID)
    assert config["review_channel"] == TEST_CHANNEL_ID


async def test_set_config_field_rejects_unmapped_field_name(session: AsyncSession) -> None:
    """A field name with no corresponding Guild column (typo, or a future
    command field added without updating _FIELD_TO_COLUMN/the model) must
    raise loudly, not silently no-op like the pre-fix review_channel bug.
    """
    deps = _make_deps(session)
    with pytest.raises(AttributeError):
        await deps.set_config_field(GUILD_ID, "not_a_real_field", "x")
