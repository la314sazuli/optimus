"""Slash command schema as pure data, plus a hikari builder adapter.

The command tree (names, descriptions, options, and the *server-side* permission
each command requires) is declared as plain dataclasses so it can be asserted on
in tests without importing hikari. :func:`build_command_builders` converts the
tree into hikari ``SlashCommandBuilder`` objects for registration.

``default_member_permissions`` is set on the builders purely as a client-side
convenience (it greys the command out for users who clearly lack access). It is
*never* the authorization boundary — see :data:`COMMAND_PERMISSIONS` and the
server-side re-check in :mod:`.service`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from optimus.services.interactions.logic import Permission

if TYPE_CHECKING:
    import hikari

# hikari OptionType numeric values (stable Discord API constants).
OPT_SUB_COMMAND = 1
OPT_STRING = 3
OPT_INTEGER = 4
OPT_BOOLEAN = 5
OPT_USER = 6
OPT_CHANNEL = 7
OPT_ROLE = 8
OPT_ATTACHMENT = 11


@dataclass(frozen=True, slots=True)
class Option:
    """One command (or subcommand) option."""

    name: str
    description: str
    type: int
    required: bool = False


@dataclass(frozen=True, slots=True)
class SubCommand:
    """A subcommand under a top-level command."""

    name: str
    description: str
    options: tuple[Option, ...] = ()


@dataclass(frozen=True, slots=True)
class Command:
    """A top-level slash command.

    ``required_permission`` is the permission the *service* enforces server-side
    for every (sub)command under it; ``None`` means any guild member may run it
    (e.g. ``/appeal``, ``/forget_me``). ``guild_only`` commands are refused in
    DMs even before the permission check.
    """

    name: str
    description: str
    required_permission: Permission | None = None
    guild_only: bool = True
    options: tuple[Option, ...] = ()
    subcommands: tuple[SubCommand, ...] = ()


COMMANDS: tuple[Command, ...] = (
    Command(
        name="scamhash",
        description="Manage this server's known scam-image hashes.",
        required_permission=Permission.MANAGE_GUILD,
        subcommands=(
            SubCommand(
                name="add",
                description="Add a scam hash from an image or a hex hash value.",
                options=(
                    Option("image", "An image to hash and add.", OPT_ATTACHMENT),
                    Option("phash", "Perceptual hash (hex) if not adding an image.", OPT_STRING),
                    Option("dhash", "Difference hash (hex).", OPT_STRING),
                    Option("whash", "Wavelet hash (hex).", OPT_STRING),
                ),
            ),
            SubCommand(
                name="remove",
                description="Remove a scam hash by its id.",
                options=(Option("hash_id", "The hash id to remove.", OPT_STRING, required=True),),
            ),
            SubCommand(name="list", description="List this server's scam hashes."),
            SubCommand(
                name="import",
                description="Import scam hashes from an attached JSON file.",
                options=(
                    Option("file", "A JSON export file.", OPT_ATTACHMENT, required=True),
                ),
            ),
            SubCommand(name="export", description="Export this server's scam hashes as JSON."),
        ),
    ),
    Command(
        name="config",
        description="View or change this server's configuration.",
        required_permission=Permission.MANAGE_GUILD,
        subcommands=(
            SubCommand(name="view", description="Show the current configuration."),
            SubCommand(
                name="set",
                description="Set a configuration field.",
                options=(
                    Option("field", "The field to set.", OPT_STRING, required=True),
                    Option("value", "The new value.", OPT_STRING, required=True),
                ),
            ),
        ),
    ),
    Command(
        name="stats",
        description="Show detection statistics for this server.",
        required_permission=Permission.MANAGE_GUILD,
    ),
    Command(
        name="submit_global",
        description="Submit a confirmed scam hash to the shared global database.",
        required_permission=Permission.MANAGE_GUILD,
        options=(Option("hash_id", "The local hash id to submit.", OPT_STRING, required=True),),
    ),
    Command(
        name="delete_server_data",
        description="Permanently delete ALL of this server's data (GDPR).",
        required_permission=Permission.ADMINISTRATOR,
    ),
    Command(
        name="forget_me",
        description="Erase your data and opt out of all processing.",
        required_permission=None,
        guild_only=False,
    ),
    Command(
        name="appeal",
        description="Appeal your most recent detection in this server.",
        required_permission=None,
    ),
)


#: Map of every command/subcommand path to the permission the service enforces.
COMMAND_PERMISSIONS: dict[str, Permission | None] = {}
for _cmd in COMMANDS:
    COMMAND_PERMISSIONS[_cmd.name] = _cmd.required_permission


def required_permission(command_name: str) -> Permission | None:
    """Return the server-side permission required for ``command_name``."""
    return COMMAND_PERMISSIONS.get(command_name)


def build_command_builders() -> list[hikari.api.SlashCommandBuilder]:
    """Build hikari ``SlashCommandBuilder`` objects for global registration."""
    import hikari

    builders: list[hikari.api.SlashCommandBuilder] = []
    for cmd in COMMANDS:
        builder = hikari.impl.SlashCommandBuilder(cmd.name, cmd.description)
        if cmd.required_permission is not None:
            builder.set_default_member_permissions(int(cmd.required_permission))
        if cmd.guild_only:
            builder.set_context_types([hikari.ApplicationContextType.GUILD])
        else:
            builder.set_context_types(
                [
                    hikari.ApplicationContextType.GUILD,
                    hikari.ApplicationContextType.BOT_DM,
                ]
            )
        for sub in cmd.subcommands:
            builder.add_option(
                hikari.CommandOption(
                    type=hikari.OptionType.SUB_COMMAND,
                    name=sub.name,
                    description=sub.description,
                    options=[_to_option(o) for o in sub.options],
                )
            )
        for opt in cmd.options:
            builder.add_option(_to_option(opt))
        builders.append(builder)
    return builders


def _to_option(option: Option) -> hikari.CommandOption:
    import hikari

    return hikari.CommandOption(
        type=hikari.OptionType(option.type),
        name=option.name,
        description=option.description,
        is_required=option.required,
    )
