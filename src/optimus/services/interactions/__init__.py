"""Slash command and component (button) interaction handling.

The pure decision logic (permission checks, option/import validation, config
coercion, custom-id parsing) lives in :mod:`.logic`; the slash command schema
in :mod:`.commands`; and the hikari/REST/DB runtime glue in :mod:`.service`.
"""

from __future__ import annotations

from optimus.services.interactions.logic import (
    CommandError,
    ComponentAction,
    ConfigChange,
    ImportHash,
    InteractionRejected,
    Permission,
    build_export,
    decode_component_id,
    encode_component_id,
    has_permission,
    parse_hash_hex,
    validate_config_set,
    validate_import,
)

__all__ = [
    "CommandError",
    "ComponentAction",
    "ConfigChange",
    "ImportHash",
    "InteractionRejected",
    "Permission",
    "build_export",
    "decode_component_id",
    "encode_component_id",
    "has_permission",
    "parse_hash_hex",
    "validate_config_set",
    "validate_import",
]
