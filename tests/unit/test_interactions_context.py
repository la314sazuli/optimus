"""Tests for adapting hikari command interactions into handler context."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from optimus.services.interactions.service import to_context


@dataclass(frozen=True)
class CommandOption:
    name: str
    type: int
    value: Any = None
    options: list[CommandOption] | None = None


def interaction(command_name: str, options: list[CommandOption]) -> SimpleNamespace:
    return SimpleNamespace(
        guild_id=123,
        user=SimpleNamespace(id=456),
        member=SimpleNamespace(permissions=8),
        command_name=command_name,
        options=options,
        locale="en-US",
    )


def test_to_context_preserves_direct_command_options() -> None:
    ctx = to_context(
        interaction(
            "submit_global",
            [CommandOption(name="hash_id", type=3, value="hash-123")],
        )
    )

    assert ctx.subcommand is None
    assert ctx.options == {"hash_id": "hash-123"}


def test_to_context_flattens_subcommand_options() -> None:
    ctx = to_context(
        interaction(
            "config",
            [
                CommandOption(
                    name="set",
                    type=1,
                    options=[
                        CommandOption(name="field", type=3, value="sensitivity"),
                        CommandOption(name="value", type=3, value="strict"),
                    ],
                )
            ],
        )
    )

    assert ctx.subcommand == "set"
    assert ctx.options == {"field": "sensitivity", "value": "strict"}


def test_to_context_handles_subcommand_without_options() -> None:
    ctx = to_context(
        interaction(
            "config",
            [CommandOption(name="view", type=1, options=[])],
        )
    )

    assert ctx.subcommand == "view"
    assert ctx.options == {}


def test_to_context_handles_subcommand_with_none_options() -> None:
    """Regression test: Discord sends ``options=None`` (not ``[]``) for a
    parameterless subcommand like ``/config view`` or ``/scamhash list``.

    This is indistinguishable in shape from a leaf parameter option that also
    has ``options=None`` (only ``value`` set), which previously caused the
    subcommand-descent loop to stop early and leave ``ctx.subcommand`` as
    ``None`` — silently misrouting ``/config view`` into the ``set`` branch
    and crashing with ``KeyError: 'field'``, and misrouting ``/scamhash list``
    into an ``UNKNOWN_FIELD`` rejection. The loop must discriminate on
    ``type`` (``SUB_COMMAND``/``SUB_COMMAND_GROUP``), not on whether
    ``options`` happens to be empty or ``None``.
    """
    ctx = to_context(
        interaction(
            "config",
            [CommandOption(name="view", type=1, options=None)],
        )
    )

    assert ctx.subcommand == "view"
    assert ctx.options == {}


def test_to_context_handles_scamhash_list_with_none_options() -> None:
    """Same regression as above, exercised through the exact production
    shape reported for ``/scamhash list``."""
    ctx = to_context(
        interaction(
            "scamhash",
            [CommandOption(name="list", type=1, options=None)],
        )
    )

    assert ctx.subcommand == "list"
    assert ctx.options == {}
    assert ctx.command == "scamhash"


def test_to_context_descends_through_subcommand_group() -> None:
    ctx = to_context(
        interaction(
            "config",
            [
                CommandOption(
                    name="moderation",
                    type=2,
                    options=[CommandOption(name="view", type=1, options=[])],
                )
            ],
        )
    )

    assert ctx.subcommand == "view"
    assert ctx.options == {}


def test_to_context_descends_through_subcommand_group_to_parameterless_leaf() -> None:
    """Same regression, one level deeper through a SUB_COMMAND_GROUP, with
    the terminal leaf subcommand's ``options`` as ``None`` (Discord's real
    wire shape) rather than ``[]``."""
    ctx = to_context(
        interaction(
            "config",
            [
                CommandOption(
                    name="moderation",
                    type=2,
                    options=[CommandOption(name="view", type=1, options=None)],
                )
            ],
        )
    )

    assert ctx.subcommand == "view"
    assert ctx.options == {}
