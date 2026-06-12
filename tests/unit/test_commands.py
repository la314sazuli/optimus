"""Tests for the slash-command tree -> hikari builder adapter."""

from __future__ import annotations

import hikari

from optimus.services.interactions.commands import (
    COMMANDS,
    build_command_builders,
    required_permission,
)
from optimus.services.interactions.logic import Permission


def test_build_command_builders_covers_every_declared_command() -> None:
    builders = build_command_builders()
    assert {b.name for b in builders} == {c.name for c in COMMANDS}


def test_builder_sets_permissions_and_guild_only_context() -> None:
    builders = {b.name: b for b in build_command_builders()}

    scamhash = builders["scamhash"]
    assert scamhash.default_member_permissions == int(Permission.MANAGE_GUILD)
    assert hikari.ApplicationContextType.GUILD in scamhash.context_types
    assert hikari.ApplicationContextType.BOT_DM not in scamhash.context_types

    # /forget_me is permission-free and usable in DMs.
    forget = builders["forget_me"]
    assert hikari.ApplicationContextType.BOT_DM in forget.context_types


def test_builder_expands_subcommands_and_their_options() -> None:
    scamhash = next(b for b in build_command_builders() if b.name == "scamhash")
    subs = {opt.name: opt for opt in scamhash.options}
    assert "add" in subs
    assert subs["add"].type == hikari.OptionType.SUB_COMMAND
    add_opts = {o.name for o in (subs["add"].options or [])}
    assert {"image", "phash", "dhash", "whash"} <= add_opts


def test_builder_carries_required_flag_on_top_level_options() -> None:
    submit = next(b for b in build_command_builders() if b.name == "submit_global")
    hash_opt = next(o for o in submit.options if o.name == "hash_id")
    assert hash_opt.is_required is True


def test_required_permission_lookup() -> None:
    assert required_permission("scamhash") is Permission.MANAGE_GUILD
    assert required_permission("delete_server_data") is Permission.ADMINISTRATOR
    assert required_permission("appeal") is None
    assert required_permission("does_not_exist") is None
