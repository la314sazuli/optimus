"""Pure privilege-boundary checks for moderation actions.

Before any punitive action is taken against an uploader, the target must clear
these guardrails. They are intentionally conservative: when in doubt the action
is downgraded to a report so a human decides. Keeping them pure (no Discord
client) makes the full refusal matrix unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BoundaryRefusal(StrEnum):
    """Why a punitive action was refused (and downgraded to report-only)."""

    GUILD_OWNER = "guild_owner"
    ADMINISTRATOR = "administrator"
    ROLE_HIERARCHY = "role_hierarchy"
    NOT_IN_GUILD = "not_in_guild"
    SELF = "self"


@dataclass(frozen=True, slots=True)
class TargetContext:
    """The facts about a target needed to evaluate privilege boundaries."""

    user_id: int
    guild_owner_id: int
    bot_user_id: int
    #: Whether the target currently holds the Administrator permission.
    is_administrator: bool
    #: The target's highest role position (Discord role hierarchy).
    top_role_position: int
    #: The bot's own highest role position.
    bot_top_role_position: int
    #: Whether the event originated in a guild (False for DMs).
    in_guild: bool = True


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    """Whether a punitive action is permitted, with the refusal reason if not."""

    allowed: bool
    refusal: BoundaryRefusal | None = None


def check_target(ctx: TargetContext) -> BoundaryResult:
    """Return whether a punitive action against ``ctx`` is permitted.

    Refusals are returned (not raised) so the caller can downgrade gracefully.
    The order reflects severity: structural impossibilities first, then
    privilege, then hierarchy.
    """
    if not ctx.in_guild:
        return BoundaryResult(False, BoundaryRefusal.NOT_IN_GUILD)
    if ctx.user_id == ctx.bot_user_id:
        return BoundaryResult(False, BoundaryRefusal.SELF)
    if ctx.user_id == ctx.guild_owner_id:
        return BoundaryResult(False, BoundaryRefusal.GUILD_OWNER)
    if ctx.is_administrator:
        return BoundaryResult(False, BoundaryRefusal.ADMINISTRATOR)
    # Discord only lets a member act on targets strictly below them.
    if ctx.top_role_position >= ctx.bot_top_role_position:
        return BoundaryResult(False, BoundaryRefusal.ROLE_HIERARCHY)
    return BoundaryResult(True)
