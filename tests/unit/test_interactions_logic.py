"""Pure interaction logic: permissions, hash parsing, import validation, config."""

from __future__ import annotations

import contextlib
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from optimus.services.interactions.commands import (
    COMMAND_PERMISSIONS,
    COMMANDS,
    MESSAGE_COMMANDS,
    REVIEW_MESSAGE_COMMAND,
    build_context_menu_command_builders,
    required_permission,
)
from optimus.services.interactions.logic import (
    MAX_IMPORT_BYTES,
    MAX_IMPORT_HASHES,
    MAX_UINT64,
    CommandError,
    ComponentAction,
    InteractionRejected,
    Permission,
    build_export,
    decode_component_id,
    encode_component_id,
    has_permission,
    parse_hash_hex,
    parse_message_reference,
    validate_config_set,
    validate_import,
)

# --- permission matrix ---------------------------------------------------------


def test_administrator_implies_manage_guild() -> None:
    assert has_permission(int(Permission.ADMINISTRATOR), Permission.MANAGE_GUILD) is True


def test_manage_guild_does_not_imply_administrator() -> None:
    assert has_permission(int(Permission.MANAGE_GUILD), Permission.ADMINISTRATOR) is False


def test_no_permissions_denied() -> None:
    assert has_permission(0, Permission.MANAGE_GUILD) is False


def test_extra_unrelated_bits_ignored() -> None:
    # Bit for some other permission set, but not MANAGE_GUILD.
    assert has_permission(1 << 10, Permission.MANAGE_GUILD) is False


def test_command_permission_table_matches_declarations() -> None:
    for cmd in COMMANDS:
        assert required_permission(cmd.name) == cmd.required_permission


def test_scamhash_requires_manage_guild() -> None:
    assert required_permission("scamhash") == Permission.MANAGE_GUILD


def test_delete_server_requires_administrator() -> None:
    assert required_permission("delete_server_data") == Permission.ADMINISTRATOR


def test_appeal_and_forget_me_require_no_permission() -> None:
    assert required_permission("appeal") is None
    assert required_permission("forget_me") is None


# --- context-menu commands ("Review as scam") -----------------------------------


def test_review_message_requires_manage_guild() -> None:
    assert COMMAND_PERMISSIONS[REVIEW_MESSAGE_COMMAND] == Permission.MANAGE_GUILD
    assert required_permission(REVIEW_MESSAGE_COMMAND) == Permission.MANAGE_GUILD


def test_build_context_menu_command_builders_has_review_as_scam() -> None:
    builders = build_context_menu_command_builders()
    assert len(builders) == len(MESSAGE_COMMANDS)
    names = [b.name for b in builders]
    assert "Review as scam" in names


def test_build_context_menu_command_builders_sets_manage_guild_permission() -> None:
    import hikari

    builders = build_context_menu_command_builders()
    review = next(b for b in builders if b.name == "Review as scam")
    assert review.type is hikari.CommandType.MESSAGE
    assert review.default_member_permissions == int(Permission.MANAGE_GUILD)


# --- hash hex parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0x1", 1), ("ff", 255), ("0xDEADBEEF", 0xDEADBEEF), ("ffffffffffffffff", MAX_UINT64)],
)
def test_parse_hash_hex_valid(raw: str, expected: int) -> None:
    assert parse_hash_hex(raw) == expected


@pytest.mark.parametrize("raw", ["", "0x", "xyz", "g123", "1" * 17, "0xffffffffffffffff0"])
def test_parse_hash_hex_invalid(raw: str) -> None:
    with pytest.raises(InteractionRejected) as exc:
        parse_hash_hex(raw)
    assert exc.value.reason is CommandError.INVALID_HEX


# --- import validation ---------------------------------------------------------


def _doc(hashes: list[dict[str, object]], version: int = 1) -> str:
    return json.dumps({"version": version, "hashes": hashes})


def test_validate_import_accepts_well_formed() -> None:
    entries = validate_import(_doc([{"phash": 1, "dhash": 2, "whash": 3, "note": "x"}]))
    assert len(entries) == 1
    assert entries[0].note == "x"


def test_validate_import_note_optional() -> None:
    entries = validate_import(_doc([{"phash": 1, "dhash": 2, "whash": 3}]))
    assert entries[0].note is None


@pytest.mark.parametrize(
    "doc",
    [
        "{not json",
        json.dumps([]),  # not an object
        json.dumps({"version": 2, "hashes": [{"phash": 1, "dhash": 2, "whash": 3}]}),
        json.dumps({"version": 1, "hashes": []}),  # empty
        json.dumps({"version": 1}),  # missing hashes
        json.dumps({"version": 1, "hashes": [{"phash": 1, "dhash": 2}]}),  # missing whash
        json.dumps({"version": 1, "hashes": [{"phash": 1, "dhash": 2, "whash": 3, "x": 1}]}),
        json.dumps({"version": 1, "hashes": [{"phash": -1, "dhash": 2, "whash": 3}]}),
        json.dumps({"version": 1, "hashes": [{"phash": True, "dhash": 2, "whash": 3}]}),
        json.dumps({"version": 1, "hashes": [{"phash": 1.5, "dhash": 2, "whash": 3}]}),
        json.dumps({"version": 1, "extra": 1, "hashes": [{"phash": 1, "dhash": 2, "whash": 3}]}),
    ],
)
def test_validate_import_rejects_invalid(doc: str) -> None:
    with pytest.raises(InteractionRejected) as exc:
        validate_import(doc)
    assert exc.value.reason in (CommandError.IMPORT_INVALID, CommandError.IMPORT_TOO_LARGE)


def test_validate_import_rejects_over_uint64() -> None:
    with pytest.raises(InteractionRejected):
        validate_import(_doc([{"phash": MAX_UINT64 + 1, "dhash": 2, "whash": 3}]))


def test_validate_import_rejects_too_many() -> None:
    big = [{"phash": i, "dhash": 0, "whash": 0} for i in range(MAX_IMPORT_HASHES + 1)]
    with pytest.raises(InteractionRejected) as exc:
        validate_import(_doc(big))
    assert exc.value.reason is CommandError.IMPORT_TOO_LARGE


def test_validate_import_rejects_oversized_blob() -> None:
    blob = b'{"version":1,"hashes":[' + b"0" * MAX_IMPORT_BYTES + b"]}"
    with pytest.raises(InteractionRejected) as exc:
        validate_import(blob)
    assert exc.value.reason is CommandError.IMPORT_TOO_LARGE


def test_validate_import_rejects_long_note() -> None:
    with pytest.raises(InteractionRejected):
        validate_import(_doc([{"phash": 1, "dhash": 2, "whash": 3, "note": "x" * 257}]))


def test_export_roundtrips_through_import() -> None:
    entries = validate_import(_doc([{"phash": 10, "dhash": 20, "whash": 30, "note": "n"}]))
    exported = build_export(entries)
    reparsed = validate_import(exported)
    assert reparsed[0].phash == 10
    assert reparsed[0].dhash == 20
    assert reparsed[0].whash == 30


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "phash": st.integers(min_value=0, max_value=MAX_UINT64),
                "dhash": st.integers(min_value=0, max_value=MAX_UINT64),
                "whash": st.integers(min_value=0, max_value=MAX_UINT64),
            }
        ),
        min_size=1,
        max_size=20,
    )
)
def test_validate_import_fuzz_valid_documents(hashes: list[dict[str, int]]) -> None:
    entries = validate_import(_doc(hashes))
    assert len(entries) == len(hashes)


@given(st.text(max_size=200))
def test_validate_import_never_crashes_on_text(blob: str) -> None:
    # Arbitrary text must either parse or raise InteractionRejected, never else.
    with contextlib.suppress(InteractionRejected):
        validate_import(blob)


# --- config set ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("sensitivity", "strict", "strict"),
        ("action_policy", "delete_ban", "delete_ban"),
        ("mod_queue_threshold", "0.7", 0.7),
        ("retention_days", "30", 30),
        ("ban_purge_hours", "0", 0),
        ("ban_purge_hours", "24", 24),
        ("ban_purge_hours", "168", 168),
        ("locale", "sr", "sr"),
        ("optin_global_db", "true", True),
        ("optin_scan_bots", "off", False),
        ("review_channel", "<#1402357722430570498>", 1402357722430570498),
        ("review_channel", "1402357722430570498", 1402357722430570498),
        ("review_channel", "none", None),
        ("review_channel", "off", None),
        ("review_channel", "clear", None),
        ("review_channel", "0", None),
    ],
)
def test_validate_config_set_valid(field: str, value: str, expected: object) -> None:
    assert validate_config_set(field, value).value == expected


def test_validate_config_set_unknown_field() -> None:
    with pytest.raises(InteractionRejected) as exc:
        validate_config_set("nope", "x")
    assert exc.value.reason is CommandError.UNKNOWN_FIELD


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sensitivity", "nuclear"),
        ("action_policy", "explode"),
        ("mod_queue_threshold", "2.0"),
        ("mod_queue_threshold", "abc"),
        ("retention_days", "0"),
        ("retention_days", "9999"),
        ("ban_purge_hours", "-1"),
        ("ban_purge_hours", "169"),
        ("ban_purge_hours", "abc"),
        ("locale", "xx"),
        ("optin_global_db", "maybe"),
        ("review_channel", "not-a-channel"),
        ("review_channel", "<#not-numeric>"),
        ("review_channel", "-1"),
        ("review_channel", str(2**64)),
    ],
)
def test_validate_config_set_invalid_value(field: str, value: str) -> None:
    with pytest.raises(InteractionRejected) as exc:
        validate_config_set(field, value)
    assert exc.value.reason is CommandError.INVALID_VALUE


# --- component custom-id round trip --------------------------------------------


@pytest.mark.parametrize("action", list(ComponentAction))
def test_component_id_roundtrip(action: ComponentAction) -> None:
    encoded = encode_component_id(action, 4242)
    parsed = decode_component_id(encoded)
    assert parsed is not None
    assert parsed.action is action
    assert parsed.ref_id == 4242


@pytest.mark.parametrize(
    "bad",
    ["", "x:y:z:1", "om:v1:bogus:1", "om:v1:appeal_open:notint", "om:v2:appeal_open:1"],
)
def test_decode_component_id_rejects_foreign(bad: str) -> None:
    assert decode_component_id(bad) is None


# --- parse_message_reference ----------------------------------------------------


@pytest.mark.parametrize(
    "link",
    [
        "https://discord.com/channels/1402357722430570498/1408060752371257535/1517941473855799567",
        "https://discordapp.com/channels/1402357722430570498/1408060752371257535/1517941473855799567",
        "https://ptb.discord.com/channels/1402357722430570498/1408060752371257535/1517941473855799567",
        "https://canary.discord.com/channels/1402357722430570498/1408060752371257535/1517941473855799567",
        (
            "  https://discord.com/channels/1402357722430570498"
            "/1408060752371257535/1517941473855799567  "
        ),
    ],
)
def test_parse_message_reference_link_forms(link: str) -> None:
    channel_id, message_id = parse_message_reference(link)
    assert channel_id == 1408060752371257535
    assert message_id == 1517941473855799567


def test_parse_message_reference_bare_id_has_no_channel() -> None:
    channel_id, message_id = parse_message_reference("1517941473855799567")
    assert channel_id is None
    assert message_id == 1517941473855799567


def test_parse_message_reference_bare_id_strips_whitespace() -> None:
    channel_id, message_id = parse_message_reference("  1517941473855799567  ")
    assert channel_id is None
    assert message_id == 1517941473855799567


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "not-a-message",
        "https://discord.com/channels/1/2/notanumber",
        "https://example.com/channels/1/2/3",
        str(MAX_UINT64 + 1),
        "-1",
    ],
)
def test_parse_message_reference_rejects_invalid(bad: str) -> None:
    with pytest.raises(InteractionRejected) as exc:
        parse_message_reference(bad)
    assert exc.value.reason is CommandError.MESSAGE_NOT_FOUND
