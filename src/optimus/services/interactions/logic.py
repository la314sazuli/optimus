"""Pure, runtime-free logic for slash commands and component interactions.

Everything that decides *whether* an interaction is allowed and *what* it means
lives here, free of hikari and any I/O, so the permission matrix, option
parsing, import validation, and config coercion are exhaustively unit-testable.
The thin hikari/REST/DB glue lives in :mod:`optimus.services.interactions.service`.

Permission rule (defense in depth): a component's ``default_member_permissions``
is only a client-side hint and must never be trusted. Every state-changing
interaction is re-checked server-side against the invoking member's *effective*
permissions (:func:`has_permission`) before any side effect runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntFlag, StrEnum
from typing import Any

from optimus.core.config import Sensitivity

#: Inclusive bound for an unsigned 64-bit perceptual hash.
MAX_UINT64 = (1 << 64) - 1

#: Maximum number of hashes accepted in a single ``/scamhash import`` payload.
MAX_IMPORT_HASHES = 1000

#: Largest raw import document we will even attempt to parse (bytes).
MAX_IMPORT_BYTES = 1 << 20

#: Schema version this build emits and accepts for import/export.
IMPORT_SCHEMA_VERSION = 1


class Permission(IntFlag):
    """The subset of Discord guild permissions this bot enforces.

    Values mirror Discord's permission bitfield so a raw permissions integer
    from an interaction can be checked directly.
    """

    MANAGE_GUILD = 1 << 5
    ADMINISTRATOR = 1 << 3


def has_permission(member_permissions: int, required: Permission) -> bool:
    """Whether ``member_permissions`` satisfies ``required``.

    ``ADMINISTRATOR`` implies every other permission, matching Discord. The
    integer is the member's *effective* permission set as resolved by the
    gateway/REST layer (role permissions OR'd together, owner short-circuited),
    never the command's ``default_member_permissions`` hint.
    """
    perms = Permission(member_permissions & (Permission.MANAGE_GUILD | Permission.ADMINISTRATOR))
    if Permission.ADMINISTRATOR in perms:
        return True
    return required in perms


class CommandError(StrEnum):
    """A machine-readable reason an interaction was rejected.

    Each value is also the i18n key suffix under ``command.`` used to localize
    the ephemeral error shown to the invoker.
    """

    NO_PERMISSION = "no_permission"
    GUILD_ONLY = "guild_only"
    RATE_LIMITED = "rate_limited"
    INVALID_HEX = "invalid_hex"
    IMPORT_INVALID = "import_invalid"
    IMPORT_TOO_LARGE = "import_too_large"
    UNKNOWN_FIELD = "config_unknown_field"
    INVALID_VALUE = "config_invalid_value"
    BELOW_THRESHOLD = "submit_global_below_threshold"


class InteractionRejected(Exception):  # noqa: N818 - control-flow signal, not an error
    """Raised by pure validators when an interaction must be refused."""

    def __init__(self, reason: CommandError) -> None:
        super().__init__(reason.value)
        self.reason = reason


def parse_hash_hex(raw: str) -> int:
    """Parse a user-supplied 64-bit hash, accepting ``0x``-prefixed or bare hex.

    Raises :class:`InteractionRejected` (``INVALID_HEX``) on anything that is not
    a hex string in ``[0, 2**64)``.
    """
    text = raw.strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if not text or len(text) > 16 or any(c not in "0123456789abcdef" for c in text):
        raise InteractionRejected(CommandError.INVALID_HEX)
    value = int(text, 16)
    if not 0 <= value <= MAX_UINT64:
        raise InteractionRejected(CommandError.INVALID_HEX)
    return value


@dataclass(frozen=True, slots=True)
class ImportHash:
    """One validated hash entry from a ``/scamhash import`` document."""

    phash: int
    dhash: int
    whash: int
    note: str | None = None


def _coerce_uint64(value: Any) -> int:
    """Coerce a JSON scalar to an unsigned 64-bit int or reject the import."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InteractionRejected(CommandError.IMPORT_INVALID)
    if not 0 <= value <= MAX_UINT64:
        raise InteractionRejected(CommandError.IMPORT_INVALID)
    return value


def validate_import(raw: str | bytes) -> list[ImportHash]:
    """Strictly validate a ``/scamhash import`` document; return parsed entries.

    The accepted schema is ``{"version": 1, "hashes": [{phash, dhash, whash,
    note?}, ...]}``. Unknown keys, wrong types, an unsupported version, out of
    range hashes, an over-long note, or too many/zero entries are all rejected.
    The byte cap is enforced before parsing to bound the work done on hostile
    input.
    """
    blob = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(blob) > MAX_IMPORT_BYTES:
        raise InteractionRejected(CommandError.IMPORT_TOO_LARGE)
    try:
        doc = json.loads(blob)
    except (json.JSONDecodeError, ValueError) as exc:
        raise InteractionRejected(CommandError.IMPORT_INVALID) from exc
    if not isinstance(doc, dict):
        raise InteractionRejected(CommandError.IMPORT_INVALID)
    if set(doc) - {"version", "hashes"}:
        raise InteractionRejected(CommandError.IMPORT_INVALID)
    if doc.get("version") != IMPORT_SCHEMA_VERSION:
        raise InteractionRejected(CommandError.IMPORT_INVALID)
    entries = doc.get("hashes")
    if not isinstance(entries, list) or not entries:
        raise InteractionRejected(CommandError.IMPORT_INVALID)
    if len(entries) > MAX_IMPORT_HASHES:
        raise InteractionRejected(CommandError.IMPORT_TOO_LARGE)

    parsed: list[ImportHash] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise InteractionRejected(CommandError.IMPORT_INVALID)
        if set(entry) - {"phash", "dhash", "whash", "note"}:
            raise InteractionRejected(CommandError.IMPORT_INVALID)
        if not {"phash", "dhash", "whash"} <= set(entry):
            raise InteractionRejected(CommandError.IMPORT_INVALID)
        note = entry.get("note")
        if note is not None and (not isinstance(note, str) or len(note) > 256):
            raise InteractionRejected(CommandError.IMPORT_INVALID)
        parsed.append(
            ImportHash(
                phash=_coerce_uint64(entry["phash"]),
                dhash=_coerce_uint64(entry["dhash"]),
                whash=_coerce_uint64(entry["whash"]),
                note=note,
            )
        )
    return parsed


def build_export(entries: list[ImportHash]) -> str:
    """Serialize ``entries`` into a canonical export document string."""
    return json.dumps(
        {
            "version": IMPORT_SCHEMA_VERSION,
            "hashes": [
                {"phash": e.phash, "dhash": e.dhash, "whash": e.whash, "note": e.note}
                for e in entries
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class ConfigChange:
    """A validated, coerced ``/config set`` mutation ready to persist."""

    field: str
    value: Any


#: Config fields settable via ``/config set`` and their value type.
_BOOL_FIELDS = frozenset(
    {"optin_global_db", "optin_scan_bots", "optin_evidence_storage", "safe_mode"}
)
_ACTION_POLICIES = frozenset({"report_only", "delete", "delete_timeout", "delete_ban"})


def validate_config_set(field: str, raw_value: str) -> ConfigChange:
    """Validate and coerce a ``/config set <field> <value>`` pair.

    Rejects unknown fields (``UNKNOWN_FIELD``) and out-of-domain values
    (``INVALID_VALUE``). Returns the coerced Python value to write to the
    ``Guild`` row.
    """
    text = raw_value.strip()
    if field == "sensitivity":
        try:
            return ConfigChange(field, Sensitivity(text.lower()).value)
        except ValueError as exc:
            raise InteractionRejected(CommandError.INVALID_VALUE) from exc
    if field == "action_policy":
        if text.lower() not in _ACTION_POLICIES:
            raise InteractionRejected(CommandError.INVALID_VALUE)
        return ConfigChange(field, text.lower())
    if field == "mod_queue_threshold":
        try:
            value = float(text)
        except ValueError as exc:
            raise InteractionRejected(CommandError.INVALID_VALUE) from exc
        if not 0.0 <= value <= 1.0:
            raise InteractionRejected(CommandError.INVALID_VALUE)
        return ConfigChange(field, value)
    if field == "retention_days":
        try:
            days = int(text)
        except ValueError as exc:
            raise InteractionRejected(CommandError.INVALID_VALUE) from exc
        if not 1 <= days <= 365:
            raise InteractionRejected(CommandError.INVALID_VALUE)
        return ConfigChange(field, days)
    if field == "locale":
        from optimus.i18n import available_locales

        if text.lower() not in available_locales():
            raise InteractionRejected(CommandError.INVALID_VALUE)
        return ConfigChange(field, text.lower())
    if field in _BOOL_FIELDS:
        if text.lower() not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise InteractionRejected(CommandError.INVALID_VALUE)
        return ConfigChange(field, text.lower() in {"true", "1", "yes", "on"})
    raise InteractionRejected(CommandError.UNKNOWN_FIELD)


class ComponentAction(StrEnum):
    """Non-report component actions, carried in the ``om:v1`` custom-id scheme.

    These complement :class:`optimus.services.moderation.review.ReviewAction`
    (the report buttons) with the appeal lifecycle and the safe-mode resume
    control. They share the ``om:v1:<action>:<id>`` envelope.
    """

    APPEAL_OPEN = "appeal_open"
    APPEAL_APPROVE = "appeal_approve"
    APPEAL_DENY = "appeal_deny"
    SAFE_MODE_RESUME = "safe_mode_resume"
    DELETE_SERVER_CONFIRM = "delete_server_confirm"


_COMPONENT_PREFIX = "om:v1"


def encode_component_id(action: ComponentAction, ref_id: int) -> str:
    """Build an ``om:v1:<action>:<ref_id>`` custom id for a non-report control."""
    return f"{_COMPONENT_PREFIX}:{action.value}:{ref_id}"


@dataclass(frozen=True, slots=True)
class ParsedComponentId:
    """A decoded non-report component custom id."""

    action: ComponentAction
    ref_id: int


def decode_component_id(custom_id: str) -> ParsedComponentId | None:
    """Parse a non-report component custom id; ``None`` if not one of ours."""
    parts = custom_id.split(":")
    if len(parts) != 4 or f"{parts[0]}:{parts[1]}" != _COMPONENT_PREFIX:
        return None
    try:
        action = ComponentAction(parts[2])
        ref_id = int(parts[3])
    except ValueError:
        return None
    return ParsedComponentId(action=action, ref_id=ref_id)
