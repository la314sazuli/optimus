"""Validate the declared command tree against Discord's structural limits.

Discord's ``PUT /applications/{id}/commands`` bulk-overwrite endpoint validates
every command, subcommand, and option's ``name``/``description`` in a single
request and rejects the *entire batch* with a 400 if any one of them is out of
bounds (e.g. a description over 100 characters) -- see
``docs.discord.com/developers/interactions/application-commands``. Because
that failure only surfaces at runtime against the live Discord API, the whole
command tree (all top-level commands, all slash-command names) would keep
serving Discord's *last successfully registered* set, silently, until a human
happened to right-click for a specific new command and notice it missing.

These tests catch any future name/description violation locally, before it
ever reaches Discord and disables registration for every command in the bot.
"""

from __future__ import annotations

import re

import pytest

from optimus.services.interactions.commands import (
    COMMANDS,
    MESSAGE_COMMAND_LABELS,
    MESSAGE_COMMANDS,
    Command,
    Option,
    SubCommand,
)

# Discord's application-command object limits (name/description length and the
# name charset). See "Application Command Object" in Discord's docs.
_NAME_MIN, _NAME_MAX = 1, 32
_DESCRIPTION_MIN, _DESCRIPTION_MAX = 1, 100
# CHAT_INPUT command/subcommand/option names must be lowercase and may only
# contain letters, numbers, underscores, and hyphens (no spaces).
_NAME_PATTERN = re.compile(r"^[a-z0-9_-]{1,32}$")


def _assert_valid_slash_name(name: str) -> None:
    assert _NAME_MIN <= len(name) <= _NAME_MAX, f"{name!r} length out of range"
    assert _NAME_PATTERN.match(name), f"{name!r} violates Discord's CHAT_INPUT name charset"


def _assert_valid_description(description: str, *, context: str) -> None:
    assert _DESCRIPTION_MIN <= len(description) <= _DESCRIPTION_MAX, (
        f"{context} description is {len(description)} chars "
        f"(must be {_DESCRIPTION_MIN}-{_DESCRIPTION_MAX}): {description!r}"
    )


def _iter_options(container: Command | SubCommand) -> tuple[Option, ...]:
    return container.options


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: c.name)
def test_top_level_command_name_and_description_within_discord_limits(
    command: Command,
) -> None:
    _assert_valid_slash_name(command.name)
    _assert_valid_description(command.description, context=f"/{command.name}")


@pytest.mark.parametrize(
    "command",
    [c for c in COMMANDS if c.subcommands],
    ids=lambda c: c.name,
)
def test_subcommand_name_and_description_within_discord_limits(command: Command) -> None:
    for sub in command.subcommands:
        _assert_valid_slash_name(sub.name)
        _assert_valid_description(sub.description, context=f"/{command.name} {sub.name}")


@pytest.mark.parametrize(
    "command",
    [c for c in COMMANDS if c.options or c.subcommands],
    ids=lambda c: c.name,
)
def test_option_name_and_description_within_discord_limits(command: Command) -> None:
    # Top-level options (e.g. /submit_global hash_id:<...>).
    for opt in command.options:
        _assert_valid_slash_name(opt.name)
        _assert_valid_description(opt.description, context=f"/{command.name} {opt.name}")
    # Options nested one level down, under each subcommand (e.g.
    # /scamhash reviewmsg message:<...>) -- this is exactly the shape that
    # shipped a 105-char description undetected, since it's two levels deep.
    for sub in command.subcommands:
        for opt in sub.options:
            _assert_valid_slash_name(opt.name)
            _assert_valid_description(
                opt.description, context=f"/{command.name} {sub.name} {opt.name}"
            )


def test_context_menu_command_name_within_discord_limits() -> None:
    """MESSAGE/USER context-menu commands take a display name, not a slash-style one.

    Unlike CHAT_INPUT names, these can have spaces/mixed case, but still share
    the same 1-32 character length bound and must carry no description at all
    (Discord rejects a ``description`` on USER/MESSAGE commands outright).
    """
    for cmd in MESSAGE_COMMANDS:
        label = MESSAGE_COMMAND_LABELS[cmd.name]
        assert _NAME_MIN <= len(label) <= _NAME_MAX, f"{label!r} length out of range"


def test_global_command_and_context_menu_counts_within_discord_caps() -> None:
    """Discord caps global commands: 100 CHAT_INPUT, 15 USER, 15 MESSAGE.

    Nowhere near either limit today, but a silent breach here is exactly the
    same failure mode as the description-length bug: the whole bulk-overwrite
    batch gets rejected and every command silently stops updating.
    """
    assert len(COMMANDS) <= 100
    assert len(MESSAGE_COMMANDS) <= 15
