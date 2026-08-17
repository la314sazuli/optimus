"""Interaction handlers: server-side auth, audit, button auth, appeal lifecycle."""

from __future__ import annotations

from typing import Any

import pytest

from optimus.db.models import GuildHash, GuildWhitelist
from optimus.globaldb.service import GlobalHashService, SubmissionDenied
from optimus.services.interactions.attachment_hash import (
    AttachmentHashError,
    AttachmentHashes,
)
from optimus.services.interactions.handlers import (
    _CONFIG_VIEW_ORDER,
    _HASH_LIST_PREVIEW_LIMIT,
    InteractionContext,
    handle_command,
    handle_component,
    handle_review_button,
)
from optimus.services.interactions.logic import (
    CommandError,
    ComponentAction,
    InteractionRejected,
    Permission,
)
from optimus.services.interactions.service import render
from optimus.services.moderation.review import ParsedCustomId, ReviewAction

ADMIN = int(Permission.ADMINISTRATOR)
MANAGE = int(Permission.MANAGE_GUILD)
NONE = 0


class FakeDeps:
    """An in-memory :class:`InteractionDeps` that records side effects."""

    def __init__(self, **flags: Any) -> None:
        self.audits: list[tuple[int, int, str, str | None]] = []
        self.hashes: dict[str, GuildHash] = {}
        self.appeals: dict[int, dict[str, Any]] = {}
        self.reversed: list[int] = []
        self.purged: list[int] = []
        self.opted_out: list[int] = []
        self.safe_mode_disabled: list[int] = []
        self.config_set: list[tuple[str, Any]] = []
        self.resolved: list[tuple[int, bool]] = []
        self._hash_rate_ok = flags.get("hash_rate_ok", True)
        self._appeal_ok = flags.get("appeal_ok", True)
        self._recent_detection = flags.get("recent_detection", 555)
        self._owned_detections: set[int] = set(flags.get("owned_detections", {77}))
        self._next_appeal_id = 1
        self._global_service = _FakeGlobalService(flags.get("submit_error"))
        self.global_submitted: list[str] = []
        #: attachment_id -> exception to raise, or a hash_id string to return.
        self._attachment_outcomes: dict[int, Any] = flags.get("attachment_outcomes", {})
        self.confirmed_scams: list[dict[str, Any]] = []
        #: Extra get_config fields (e.g. action_policy/safe_mode) so reviewmsg
        #: outcome-reporting tests can drive each policy branch.
        self.config: dict[str, Any] = {"locale": "en", **flags.get("config", {})}

    async def add_guild_hash(self, guild_id: int, gh: GuildHash) -> GuildHash:
        self.hashes[gh.hash_id] = gh
        return gh

    async def remove_guild_hash(self, guild_id: int, hash_id: str) -> int:
        return 1 if self.hashes.pop(hash_id, None) is not None else 0

    async def list_guild_hashes(self, guild_id: int) -> list[GuildHash]:
        return list(self.hashes.values())

    async def add_whitelist(self, guild_id: int, entry: GuildWhitelist) -> GuildWhitelist:
        return entry

    async def get_config(self, guild_id: int) -> dict[str, Any]:
        return dict(self.config)

    async def set_config_field(self, guild_id: int, field: str, value: Any) -> None:
        self.config_set.append((field, value))

    async def stats_summary(self, guild_id: int) -> dict[str, Any]:
        return {"detections": 3, "hours": 24, "boots": 5, "first_boot": "2026-08-01"}

    async def opt_out_user(self, user_id: int) -> int:
        self.opted_out.append(user_id)
        return 1

    async def purge_guild(self, guild_id: int) -> int:
        self.purged.append(guild_id)
        return 7

    async def recent_detection_for(self, guild_id: int, user_id: int) -> int | None:
        return self._recent_detection

    async def detection_belongs_to(self, guild_id: int, detection_id: int, user_id: int) -> bool:
        return detection_id in self._owned_detections

    async def open_appeal(self, guild_id: int, detection_id: int, user_id: int) -> int:
        appeal_id = self._next_appeal_id
        self._next_appeal_id += 1
        self.appeals[appeal_id] = {"detection_id": detection_id, "user_id": user_id}
        return appeal_id

    async def get_appeal(self, guild_id: int, appeal_id: int) -> dict[str, Any] | None:
        return self.appeals.get(appeal_id)

    async def resolve_appeal(self, guild_id: int, appeal_id: int, *, approved: bool) -> None:
        self.resolved.append((appeal_id, approved))

    async def reverse_detection_action(self, guild_id: int, detection_id: int) -> None:
        self.reversed.append(detection_id)

    async def disable_safe_mode(self, guild_id: int) -> None:
        self.safe_mode_disabled.append(guild_id)

    async def local_hash(self, guild_id: int, hash_id: str) -> GuildHash | None:
        return self.hashes.get(hash_id)

    async def hash_rate_ok(self, user_id: int) -> bool:
        return self._hash_rate_ok

    async def appeal_cooldown_ok(self, user_id: int) -> bool:
        return self._appeal_ok

    async def audit(
        self, guild_id: int, actor_id: int, action: str, *, target: str | None = None
    ) -> None:
        self.audits.append((guild_id, actor_id, action, target))

    def global_service(self) -> GlobalHashService:
        return self._global_service  # type: ignore[return-value]

    async def compute_attachment_hashes(self, *, attachment_id: int, url: str) -> AttachmentHashes:
        outcome = self._attachment_outcomes.get(attachment_id, f"{attachment_id:016x}")
        if isinstance(outcome, Exception):
            raise outcome
        # phash is the only field _review_message's downstream store call
        # derives an id from (`f"{hashes.phash:016x}"`); stash the intended
        # hash_id string in phash's hex digits so store_attachment_hash below
        # reproduces it exactly, matching the pre-split fake's behavior.
        return AttachmentHashes(
            attachment_id=attachment_id,
            url=url,
            phash=int(outcome, 16),
            dhash=attachment_id,
            whash=attachment_id,
            ahash=0,
            mphash=0,
            mdhash=0,
            mwhash=0,
            mahash=0,
        )

    async def store_attachment_hash(
        self, guild_id: int, *, hashes: AttachmentHashes, added_by: int
    ) -> GuildHash:
        hash_id = f"{hashes.phash:016x}"
        existing = self.hashes.get(hash_id)
        if existing is not None:
            return existing
        gh = GuildHash(
            hash_id=hash_id,
            phash=hashes.phash,
            dhash=hashes.dhash,
            whash=hashes.whash,
            ahash=hashes.ahash,
            source="reviewmsg",
            added_by=added_by,
        )
        self.hashes[hash_id] = gh
        return gh

    async def submit_confirmed_scam(
        self,
        guild_id: int,
        *,
        channel_id: int,
        message_id: int,
        attachment_id: int,
        uploader_id: int,
        matched_hash_id: str,
    ) -> None:
        self.confirmed_scams.append(
            {
                "guild_id": guild_id,
                "channel_id": channel_id,
                "message_id": message_id,
                "attachment_id": attachment_id,
                "uploader_id": uploader_id,
                "matched_hash_id": matched_hash_id,
            }
        )


class _FakeGlobalService:
    """Minimal stand-in for :class:`GlobalHashService` used by submit_global."""

    def __init__(self, error: str | None) -> None:
        self._error = error
        self.submitted: list[str] = []

    async def submit(self, *, hash_id: str, **_: Any) -> None:
        if self._error is not None:
            raise SubmissionDenied(self._error)
        self.submitted.append(hash_id)


def _ctx(command: str, *, perms: int = MANAGE, guild_id: int | None = 1, **opts: Any) -> Any:
    sub = opts.pop("subcommand", None)
    return InteractionContext(
        guild_id=guild_id,
        user_id=99,
        member_permissions=perms,
        command=command,
        subcommand=sub,
        options=opts,
    )


# --- command permission matrix -------------------------------------------------


@pytest.mark.asyncio
async def test_scamhash_denied_without_manage_guild() -> None:
    with pytest.raises(InteractionRejected) as exc:
        await handle_command(_ctx("scamhash", perms=NONE, subcommand="list"), FakeDeps())
    assert exc.value.reason is CommandError.NO_PERMISSION


@pytest.mark.asyncio
async def test_scamhash_allowed_with_manage_guild() -> None:
    resp = await handle_command(_ctx("scamhash", perms=MANAGE, subcommand="list"), FakeDeps())
    assert resp.i18n_key == "command.hash_list_empty"


@pytest.mark.asyncio
async def test_admin_satisfies_manage_guild_command() -> None:
    resp = await handle_command(_ctx("scamhash", perms=ADMIN, subcommand="list"), FakeDeps())
    assert resp.i18n_key == "command.hash_list_empty"


@pytest.mark.asyncio
async def test_delete_server_denied_for_manage_guild_only() -> None:
    with pytest.raises(InteractionRejected) as exc:
        await handle_command(_ctx("delete_server_data", perms=MANAGE), FakeDeps())
    assert exc.value.reason is CommandError.NO_PERMISSION


@pytest.mark.asyncio
async def test_delete_server_allowed_for_admin() -> None:
    resp = await handle_command(_ctx("delete_server_data", perms=ADMIN), FakeDeps())
    assert resp.i18n_key == "command.delete_server_confirm"


@pytest.mark.asyncio
async def test_guild_only_command_in_dm_rejected() -> None:
    ctx = _ctx("scamhash", perms=ADMIN, guild_id=None, subcommand="list")
    with pytest.raises(InteractionRejected) as exc:
        await handle_command(ctx, FakeDeps())
    assert exc.value.reason is CommandError.GUILD_ONLY


@pytest.mark.asyncio
async def test_forget_me_allowed_in_dm_without_permission() -> None:
    deps = FakeDeps()
    resp = await handle_command(_ctx("forget_me", perms=NONE, guild_id=None), deps)
    assert resp.i18n_key == "command.forget_me_ok"
    assert deps.opted_out == [99]


# --- command side effects + audit ----------------------------------------------


@pytest.mark.asyncio
async def test_scamhash_add_audits_and_stores() -> None:
    deps = FakeDeps()
    resp = await handle_command(_ctx("scamhash", subcommand="add", phash="deadbeef"), deps)
    assert resp.i18n_key == "command.hash_added"
    assert deps.hashes
    assert deps.audits[0][2] == "scamhash.add"


@pytest.mark.asyncio
async def test_scamhash_add_rate_limited() -> None:
    deps = FakeDeps(hash_rate_ok=False)
    with pytest.raises(InteractionRejected) as exc:
        await handle_command(_ctx("scamhash", subcommand="add", phash="1"), deps)
    assert exc.value.reason is CommandError.RATE_LIMITED


@pytest.mark.asyncio
async def test_config_set_persists_and_audits() -> None:
    deps = FakeDeps()
    resp = await handle_command(
        _ctx("config", subcommand="set", field="retention_days", value="14"), deps
    )
    assert resp.i18n_key == "command.config_set_ok"
    assert deps.config_set == [("retention_days", 14)]
    assert deps.audits[0][2] == "config.set"


@pytest.mark.asyncio
async def test_config_set_review_channel_renders_as_mention() -> None:
    deps = FakeDeps()
    resp = await handle_command(
        _ctx(
            "config",
            subcommand="set",
            field="review_channel",
            value="<#1402357722430570498>",
        ),
        deps,
    )
    assert resp.i18n_key == "command.config_set_ok"
    assert deps.config_set == [("review_channel", 1402357722430570498)]
    assert resp.params["value"] == "<#1402357722430570498>"


@pytest.mark.asyncio
async def test_config_set_review_channel_clear_renders_as_none() -> None:
    deps = FakeDeps()
    resp = await handle_command(
        _ctx("config", subcommand="set", field="review_channel", value="none"), deps
    )
    assert resp.i18n_key == "command.config_set_ok"
    assert deps.config_set == [("review_channel", None)]
    assert resp.params["value"] == "none"


# --- review button auth --------------------------------------------------------


@pytest.mark.asyncio
async def test_review_button_denied_without_manage_guild() -> None:
    ctx = _ctx("", perms=NONE)
    parsed = ParsedCustomId(action=ReviewAction.CONFIRM_SCAM, detection_id=5)
    with pytest.raises(InteractionRejected) as exc:
        await handle_review_button(ctx, parsed, FakeDeps())
    assert exc.value.reason is CommandError.NO_PERMISSION


@pytest.mark.asyncio
async def test_false_positive_reverses_and_audits() -> None:
    deps = FakeDeps()
    ctx = _ctx("", perms=MANAGE)
    parsed = ParsedCustomId(action=ReviewAction.FALSE_POSITIVE, detection_id=5)
    resp = await handle_review_button(ctx, parsed, deps)
    assert resp.i18n_key == "button.marked_false_positive"
    assert deps.reversed == [5]
    assert deps.audits[0][2] == "review.false_positive"


@pytest.mark.asyncio
async def test_confirm_scam_audits() -> None:
    deps = FakeDeps()
    ctx = _ctx("", perms=MANAGE)
    parsed = ParsedCustomId(action=ReviewAction.CONFIRM_SCAM, detection_id=9)
    resp = await handle_review_button(ctx, parsed, deps)
    assert resp.i18n_key == "button.confirmed_scam"
    assert deps.audits[0] == (1, 99, "review.confirm_scam", "9")


# --- appeal lifecycle ----------------------------------------------------------


@pytest.mark.asyncio
async def test_appeal_open_via_command() -> None:
    deps = FakeDeps()
    resp = await handle_command(_ctx("appeal", perms=NONE), deps)
    assert resp.i18n_key == "command.appeal_opened"
    assert deps.appeals
    assert deps.audits[0][2] == "appeal.open"


@pytest.mark.asyncio
async def test_appeal_command_cooldown() -> None:
    deps = FakeDeps(appeal_ok=False)
    resp = await handle_command(_ctx("appeal", perms=NONE), deps)
    assert resp.i18n_key == "dm.appeal_cooldown"
    assert not deps.appeals


@pytest.mark.asyncio
async def test_appeal_command_no_detection() -> None:
    deps = FakeDeps(recent_detection=None)
    resp = await handle_command(_ctx("appeal", perms=NONE), deps)
    assert resp.i18n_key == "command.appeal_none"


@pytest.mark.asyncio
async def test_appeal_open_button() -> None:
    deps = FakeDeps()
    ctx = _ctx("", perms=NONE)
    resp = await handle_component(ctx, ComponentAction.APPEAL_OPEN, 77, deps)
    assert resp.i18n_key == "dm.appeal_submitted"
    assert deps.appeals[1]["detection_id"] == 77


@pytest.mark.asyncio
async def test_appeal_open_button_rejects_unowned_detection() -> None:
    # The detection id rides in the forgeable custom id; a user must not be able
    # to open an appeal for a detection that is not theirs (or does not exist).
    deps = FakeDeps(owned_detections=set())
    ctx = _ctx("", perms=NONE)
    resp = await handle_component(ctx, ComponentAction.APPEAL_OPEN, 999, deps)
    assert resp.i18n_key == "command.appeal_none"
    assert not deps.appeals


@pytest.mark.asyncio
async def test_appeal_open_button_rejected_in_dm() -> None:
    deps = FakeDeps()
    ctx = _ctx("", perms=NONE, guild_id=None)
    with pytest.raises(InteractionRejected) as excinfo:
        await handle_component(ctx, ComponentAction.APPEAL_OPEN, 77, deps)
    assert excinfo.value.reason is CommandError.GUILD_ONLY
    assert not deps.appeals


@pytest.mark.asyncio
async def test_appeal_approve_reverses_action() -> None:
    deps = FakeDeps()
    appeal_id = await deps.open_appeal(1, 333, 99)
    ctx = _ctx("", perms=MANAGE)
    resp = await handle_component(ctx, ComponentAction.APPEAL_APPROVE, appeal_id, deps)
    assert resp.i18n_key == "button.appeal_approved"
    assert deps.resolved == [(appeal_id, True)]
    assert deps.reversed == [333]


@pytest.mark.asyncio
async def test_appeal_deny_does_not_reverse() -> None:
    deps = FakeDeps()
    appeal_id = await deps.open_appeal(1, 333, 99)
    ctx = _ctx("", perms=MANAGE)
    resp = await handle_component(ctx, ComponentAction.APPEAL_DENY, appeal_id, deps)
    assert resp.i18n_key == "button.appeal_denied"
    assert deps.resolved == [(appeal_id, False)]
    assert deps.reversed == []


@pytest.mark.asyncio
async def test_appeal_approve_denied_without_permission() -> None:
    deps = FakeDeps()
    ctx = _ctx("", perms=NONE)
    with pytest.raises(InteractionRejected) as exc:
        await handle_component(ctx, ComponentAction.APPEAL_APPROVE, 1, deps)
    assert exc.value.reason is CommandError.NO_PERMISSION


# --- safe-mode + purge components ----------------------------------------------


@pytest.mark.asyncio
async def test_safe_mode_resume_requires_manage_guild() -> None:
    deps = FakeDeps()
    with pytest.raises(InteractionRejected):
        await handle_component(_ctx("", perms=NONE), ComponentAction.SAFE_MODE_RESUME, 1, deps)
    resp = await handle_component(_ctx("", perms=MANAGE), ComponentAction.SAFE_MODE_RESUME, 1, deps)
    assert resp.i18n_key == "button.safe_mode_resumed"
    assert deps.safe_mode_disabled == [1]


@pytest.mark.asyncio
async def test_delete_confirm_requires_admin_and_purges() -> None:
    deps = FakeDeps()
    confirm = ComponentAction.DELETE_SERVER_CONFIRM
    with pytest.raises(InteractionRejected):
        await handle_component(_ctx("", perms=MANAGE), confirm, 1, deps)
    resp = await handle_component(
        _ctx("", perms=ADMIN), ComponentAction.DELETE_SERVER_CONFIRM, 1, deps
    )
    assert resp.i18n_key == "command.delete_server_ok"
    assert deps.purged == [1]


# --- scamhash remove/list/import/export ----------------------------------------


@pytest.mark.asyncio
async def test_scamhash_remove_found_audits() -> None:
    deps = FakeDeps()
    deps.hashes["abc"] = GuildHash(
        hash_id="abc", phash=1, dhash=2, whash=3, ahash=0, source="local"
    )
    resp = await handle_command(_ctx("scamhash", subcommand="remove", hash_id="abc"), deps)
    assert resp.i18n_key == "command.hash_removed"
    assert deps.audits[0][2] == "scamhash.remove"


@pytest.mark.asyncio
async def test_scamhash_remove_not_found_does_not_audit() -> None:
    deps = FakeDeps()
    resp = await handle_command(_ctx("scamhash", subcommand="remove", hash_id="missing"), deps)
    assert resp.i18n_key == "command.hash_not_found"
    assert deps.audits == []


@pytest.mark.asyncio
async def test_scamhash_list_non_empty() -> None:
    deps = FakeDeps()
    for hash_id in ("def", "abc"):
        deps.hashes[hash_id] = GuildHash(
            hash_id=hash_id, phash=1, dhash=2, whash=3, ahash=0, source="local", added_by=42
        )
    resp = await handle_command(_ctx("scamhash", subcommand="list"), deps)
    assert resp.i18n_key == "command.hash_list_header"
    # Entries must actually be rendered (the header alone showed an empty list
    # after the colon in production), sorted, with source and adder attribution.
    assert render(resp, "en") == (
        "This server has 2 scam hash(es):\n"
        "\u2022 `abc` \u2014 local by <@42>\n"
        "\u2022 `def` \u2014 local by <@42>"
    )


@pytest.mark.asyncio
async def test_scamhash_list_truncates_to_fit_discord_reply() -> None:
    deps = FakeDeps()
    for value in range(_HASH_LIST_PREVIEW_LIMIT + 1):
        # Worst-case rows: full 64-char hash ids, max-width source column, and
        # a full-width snowflake mention on every line.
        hash_id = f"{value:064x}"
        deps.hashes[hash_id] = GuildHash(
            hash_id=hash_id,
            phash=value,
            dhash=0,
            whash=0,
            ahash=0,
            source="x" * 32,
            added_by=2**63,
        )

    resp = await handle_command(_ctx("scamhash", subcommand="list"), deps)
    message = render(resp, "en")

    assert resp.i18n_key == "command.hash_list_truncated"
    assert resp.params["count"] == _HASH_LIST_PREVIEW_LIMIT + 1
    assert f"`{_HASH_LIST_PREVIEW_LIMIT - 1:064x}`" in message
    assert f"`{_HASH_LIST_PREVIEW_LIMIT:064x}`" not in message
    assert "1 more" in message
    assert len(message) <= 2000


@pytest.mark.asyncio
async def test_scamhash_import_stores_and_dedupes() -> None:
    deps = FakeDeps()
    doc = (
        '{"version": 1, "hashes": ['
        '{"phash": 10, "dhash": 0, "whash": 0},'
        '{"phash": 10, "dhash": 0, "whash": 0},'  # duplicate phash -> skipped
        '{"phash": 20, "dhash": 0, "whash": 0}]}'
    )
    resp = await handle_command(_ctx("scamhash", subcommand="import", file=doc), deps)
    assert resp.i18n_key == "command.import_ok"
    assert resp.params["added"] == 2
    assert resp.params["skipped"] == 1
    assert len(deps.hashes) == 2
    assert deps.audits[0][2] == "scamhash.import"


@pytest.mark.asyncio
async def test_scamhash_export_roundtrips() -> None:
    deps = FakeDeps()
    deps.hashes["abc"] = GuildHash(
        hash_id="abc", phash=7, dhash=8, whash=9, ahash=0, source="local"
    )
    resp = await handle_command(_ctx("scamhash", subcommand="export"), deps)
    assert resp.i18n_key == "command.export_ok"
    assert resp.params["count"] == 1
    assert resp.attachment is not None and '"phash":7' in resp.attachment


# --- config view + stats -------------------------------------------------------


@pytest.mark.asyncio
async def test_config_view() -> None:
    resp = await handle_command(_ctx("config", subcommand="view"), FakeDeps())
    assert resp.i18n_key == "command.config_view_header"
    # FakeDeps.get_config's default {"locale": "en"} still renders a summary line.
    assert "**locale**: `en`" in resp.params["summary"]


@pytest.mark.asyncio
async def test_config_view_renders_review_channel_as_mention() -> None:
    deps = FakeDeps()
    deps.get_config = _fake_get_config_with_channel  # type: ignore[method-assign]
    resp = await handle_command(_ctx("config", subcommand="view"), deps)
    assert "<#1402357722430570498>" in resp.params["summary"]


async def _fake_get_config_with_channel(guild_id: int) -> dict[str, Any]:
    return {"locale": "en", "review_channel": 1402357722430570498}


@pytest.mark.asyncio
async def test_config_view_no_row_shows_defaults_message() -> None:
    deps = FakeDeps()
    deps.get_config = _fake_get_config_empty  # type: ignore[method-assign]
    resp = await handle_command(_ctx("config", subcommand="view"), deps)
    assert "defaults are in effect" in resp.params["summary"]


async def _fake_get_config_empty(guild_id: int) -> dict[str, Any]:
    return {}


@pytest.mark.parametrize("field", _CONFIG_VIEW_ORDER)
def test_every_config_view_field_is_settable_under_the_same_name(field: str) -> None:
    """Every field /config view can display must be settable via /config set
    under that exact same name -- guards against the DB-column-name leak this
    regressed as for "review_channel" (view said "review_channel_id").

    Uses each field's already-valid value from FakeDeps' default config shape
    so this only checks *name* recognition, not value validation (that's
    covered by test_validate_config_set_valid/_invalid_value already).
    """
    from optimus.services.interactions.logic import validate_config_set

    sample_values = {
        "sensitivity": "balanced",
        "action_policy": "report_only",
        "mod_queue_threshold": "0.5",
        "review_channel": "none",
        "ban_purge_hours": "24",
        "safe_mode": "false",
        "retention_days": "30",
        "locale": "en",
        "optin_global_db": "false",
        "optin_scan_bots": "false",
        "optin_evidence_storage": "false",
    }
    assert field in sample_values, f"add a sample value for new config field {field!r}"
    # Must not raise InteractionRejected(UNKNOWN_FIELD) -- i.e. the name itself
    # is recognized. A ValueError here would mean the sample value is wrong,
    # which is a test bug, not a production one.
    validate_config_set(field, sample_values[field])


@pytest.mark.asyncio
async def test_config_set_field_name_matches_config_view_field_name() -> None:
    """Regression test: /config set field:<X> and /config view must agree on
    the name <X> for every field, so a name copied from one always works
    verbatim in the other. This previously drifted for the review channel:
    /config set accepted "review_channel" but /config view rendered it back
    as "review_channel_id" (the raw DB column name), so a user reading
    /config view's output and pasting it into /config set got an "Unknown
    configuration field" rejection.
    """
    store: dict[str, Any] = {"locale": "en"}
    deps = FakeDeps()

    async def fake_get_config(guild_id: int) -> dict[str, Any]:
        return dict(store)

    async def fake_set_config_field(guild_id: int, field: str, value: Any) -> None:
        store[field] = value
        deps.config_set.append((field, value))

    deps.get_config = fake_get_config  # type: ignore[method-assign]
    deps.set_config_field = fake_set_config_field  # type: ignore[method-assign]

    set_resp = await handle_command(
        _ctx(
            "config",
            subcommand="set",
            field="review_channel",
            value="<#1402357722430570498>",
        ),
        deps,
    )
    assert set_resp.i18n_key == "command.config_set_ok"

    view_resp = await handle_command(_ctx("config", subcommand="view"), deps)
    assert "**review_channel**: `<#1402357722430570498>`" in view_resp.params["summary"]
    assert "review_channel_id" not in view_resp.params["summary"]


@pytest.mark.asyncio
async def test_stats_non_empty() -> None:
    resp = await handle_command(_ctx("stats"), FakeDeps())
    assert resp.i18n_key == "command.stats_header"
    assert resp.params["hours"] == 24
    assert "detections" in resp.params  # header renders a body, not a bare colon
    # The persistence canary is rendered for moderators to eyeball after deploys.
    assert render(resp, "en") == (
        "Statistics for the last 24 hours:\n"
        "\u2022 Detections: 3\n"
        "\u2022 Database: boot #5, storing data since 2026-08-01"
    )


@pytest.mark.asyncio
async def test_stats_zero_detections_still_shows_persistence_canary() -> None:
    deps = FakeDeps()

    async def _quiet(guild_id: int) -> dict[str, Any]:
        return {"detections": 0, "hours": 24, "boots": 2, "first_boot": "2026-08-01"}

    deps.stats_summary = _quiet  # type: ignore[method-assign]
    resp = await handle_command(_ctx("stats"), deps)
    # A quiet server must still surface the boot counter — that's the whole
    # point of the canary: verify persistence even when nothing was detected.
    assert resp.i18n_key == "command.stats_header"
    assert resp.params["boots"] == 2


# --- submit_global -------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_global_ok() -> None:
    deps = FakeDeps()
    deps.hashes["abc"] = GuildHash(
        hash_id="abc", phash=1, dhash=2, whash=3, ahash=0, source="local"
    )
    resp = await handle_command(_ctx("submit_global", hash_id="abc"), deps)
    assert resp.i18n_key == "command.submit_global_ok"
    assert deps._global_service.submitted == ["abc"]
    assert deps.audits[0][2] == "global.submit"


@pytest.mark.asyncio
async def test_submit_global_unknown_hash() -> None:
    deps = FakeDeps()
    resp = await handle_command(_ctx("submit_global", hash_id="nope"), deps)
    assert resp.i18n_key == "command.hash_not_found"


@pytest.mark.asyncio
async def test_submit_global_below_threshold_rejected() -> None:
    deps = FakeDeps(submit_error="below_threshold")
    deps.hashes["abc"] = GuildHash(
        hash_id="abc", phash=1, dhash=2, whash=3, ahash=0, source="local"
    )
    with pytest.raises(InteractionRejected) as exc:
        await handle_command(_ctx("submit_global", hash_id="abc"), deps)
    assert exc.value.reason is CommandError.BELOW_THRESHOLD


@pytest.mark.asyncio
async def test_submit_global_rate_limited_rejected() -> None:
    deps = FakeDeps(submit_error="rate_limited")
    deps.hashes["abc"] = GuildHash(
        hash_id="abc", phash=1, dhash=2, whash=3, ahash=0, source="local"
    )
    with pytest.raises(InteractionRejected) as exc:
        await handle_command(_ctx("submit_global", hash_id="abc"), deps)
    assert exc.value.reason is CommandError.RATE_LIMITED


# --- remaining review button actions -------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "key"),
    [
        (ReviewAction.BAN_UPLOADER, "button.uploader_banned"),
        (ReviewAction.UNBAN, "button.uploader_unbanned"),
        (ReviewAction.WHITELIST_IMAGE, "button.image_whitelisted"),
        (ReviewAction.SUBMIT_GLOBAL, "button.submitted_global"),
    ],
)
async def test_review_button_actions_audit(action: ReviewAction, key: str) -> None:
    deps = FakeDeps()
    parsed = ParsedCustomId(action=action, detection_id=5)
    resp = await handle_review_button(_ctx("", perms=MANAGE), parsed, deps)
    assert resp.i18n_key == key
    assert deps.audits[0][1] == 99


# --- /scamhash reviewmsg and the "Review as scam" context-menu entry ----------


def _review_ctx(
    *,
    command: str = "scamhash",
    subcommand: str | None = "reviewmsg",
    attachments: list[tuple[int, str]] | None = None,
    channel_id: int = 111,
    message_id: int = 222,
    author_id: int = 333,
    perms: int = MANAGE,
) -> InteractionContext:
    return InteractionContext(
        guild_id=1,
        user_id=99,
        member_permissions=perms,
        command=command,
        subcommand=subcommand,
        options={
            "channel_id": channel_id,
            "message_id": message_id,
            "author_id": author_id,
            "attachments": attachments if attachments is not None else [(1, "https://x/1.png")],
        },
    )


@pytest.mark.asyncio
async def test_reviewmsg_denied_without_manage_guild() -> None:
    with pytest.raises(InteractionRejected) as exc:
        await handle_command(_review_ctx(perms=NONE), FakeDeps())
    assert exc.value.reason is CommandError.NO_PERMISSION


@pytest.mark.asyncio
async def test_reviewmsg_rate_limited_rejected() -> None:
    deps = FakeDeps(hash_rate_ok=False)
    with pytest.raises(InteractionRejected) as exc:
        await handle_command(_review_ctx(), deps)
    assert exc.value.reason is CommandError.RATE_LIMITED


@pytest.mark.asyncio
async def test_reviewmsg_no_images_short_circuits() -> None:
    deps = FakeDeps()
    resp = await handle_command(_review_ctx(attachments=[]), deps)
    assert resp.i18n_key == "command.reviewmsg_no_images"
    assert deps.confirmed_scams == []


@pytest.mark.asyncio
async def test_reviewmsg_single_attachment_hashes_and_actions_author() -> None:
    deps = FakeDeps()
    resp = await handle_command(
        _review_ctx(attachments=[(1, "https://x/1.png")], author_id=333), deps
    )
    assert resp.i18n_key == "command.reviewmsg_result_report_only"
    assert resp.params == {"added": 1, "failed": 0, "author_id": 333, "action": "report_only"}
    assert len(deps.confirmed_scams) == 1
    assert deps.confirmed_scams[0]["uploader_id"] == 333
    assert deps.confirmed_scams[0]["attachment_id"] == 1
    assert len(deps.audits) == 1
    assert deps.audits[0][2] == "scamhash.reviewmsg"


@pytest.mark.asyncio
async def test_reviewmsg_multiple_attachments_all_succeed() -> None:
    deps = FakeDeps()
    attachments = [(1, "https://x/1.png"), (2, "https://x/2.png"), (3, "https://x/3.png")]
    resp = await handle_command(_review_ctx(attachments=attachments), deps)
    assert resp.i18n_key == "command.reviewmsg_result_report_only"
    assert resp.params["added"] == 3
    assert resp.params["failed"] == 0
    assert len(deps.confirmed_scams) == 3
    assert len(deps.hashes) == 3


@pytest.mark.asyncio
async def test_reviewmsg_some_attachments_fail_are_skipped() -> None:
    deps = FakeDeps(attachment_outcomes={2: AttachmentHashError("bad image")})
    attachments = [(1, "https://x/1.png"), (2, "https://x/2.png"), (3, "https://x/3.png")]
    resp = await handle_command(_review_ctx(attachments=attachments), deps)
    assert resp.i18n_key == "command.reviewmsg_result_report_only"
    assert resp.params["added"] == 2
    assert resp.params["failed"] == 1
    # The failed attachment never reaches the moderation pipeline.
    assert {c["attachment_id"] for c in deps.confirmed_scams} == {1, 3}


@pytest.mark.asyncio
async def test_reviewmsg_all_attachments_fail() -> None:
    deps = FakeDeps(
        attachment_outcomes={
            1: AttachmentHashError("bad"),
            2: AttachmentHashError("bad"),
        }
    )
    attachments = [(1, "https://x/1.png"), (2, "https://x/2.png")]
    resp = await handle_command(_review_ctx(attachments=attachments), deps)
    assert resp.i18n_key == "command.reviewmsg_all_failed"
    assert resp.params == {"failed": 2}
    assert deps.confirmed_scams == []


@pytest.mark.asyncio
async def test_reviewmsg_duplicate_hash_is_idempotent() -> None:
    """Re-reviewing a message (or an image already hashed by another path)
    must not raise -- ``store_attachment_hash`` returns the existing row."""
    deps = FakeDeps()
    existing = GuildHash(hash_id=f"{1:016x}", phash=1, dhash=1, whash=1, ahash=0, source="local")
    deps.hashes[existing.hash_id] = existing
    resp = await handle_command(_review_ctx(attachments=[(1, "https://x/1.png")]), deps)
    assert resp.i18n_key == "command.reviewmsg_result_report_only"
    assert resp.params["added"] == 1
    assert len(deps.confirmed_scams) == 1
    assert deps.confirmed_scams[0]["matched_hash_id"] == existing.hash_id


@pytest.mark.asyncio
async def test_reviewmsg_computes_all_hashes_before_storing_any() -> None:
    """All ``compute_attachment_hashes`` calls (network fetch + decode -- no DB)
    must happen before any ``store_attachment_hash``/``audit``/
    ``submit_confirmed_scam`` call (DB writes).

    This is the actual behavioral guarantee behind splitting
    ``hash_and_store_attachment`` into a compute phase and a store phase: a
    multi-image review must never interleave a slow network+decode call for
    one attachment with a DB write for another, because the whole handler
    runs inside one caller-managed transaction (see
    :meth:`InteractionService._run`) and a SQLite write transaction holds an
    exclusive file-level lock for as long as it stays open. Interleaving used
    to hold that lock open for the sum of every attachment's fetch+decode
    time plus every write, which could starve a concurrent writer into a
    "database is locked" error well past any reasonable retry budget.
    """
    calls: list[str] = []

    class OrderTrackingDeps(FakeDeps):
        async def compute_attachment_hashes(
            self, *, attachment_id: int, url: str
        ) -> AttachmentHashes:
            calls.append(f"compute:{attachment_id}")
            return await super().compute_attachment_hashes(attachment_id=attachment_id, url=url)

        async def store_attachment_hash(
            self, guild_id: int, *, hashes: AttachmentHashes, added_by: int
        ) -> GuildHash:
            calls.append(f"store:{hashes.attachment_id}")
            return await super().store_attachment_hash(guild_id, hashes=hashes, added_by=added_by)

        async def audit(
            self, guild_id: int, actor_id: int, action: str, *, target: str | None = None
        ) -> None:
            calls.append("audit")
            await super().audit(guild_id, actor_id, action, target=target)

        async def submit_confirmed_scam(self, guild_id: int, **kwargs: Any) -> None:
            calls.append(f"submit:{kwargs['attachment_id']}")
            await super().submit_confirmed_scam(guild_id, **kwargs)

    deps = OrderTrackingDeps()
    attachments = [(1, "https://x/1.png"), (2, "https://x/2.png"), (3, "https://x/3.png")]
    resp = await handle_command(_review_ctx(attachments=attachments), deps)

    assert resp.params["added"] == 3
    compute_calls = [c for c in calls if c.startswith("compute:")]
    other_calls = [c for c in calls if not c.startswith("compute:")]
    assert compute_calls == ["compute:1", "compute:2", "compute:3"]
    assert calls.index(other_calls[0]) > calls.index(compute_calls[-1]), (
        "a store/audit/submit call happened before all attachments were computed"
    )


@pytest.mark.asyncio
async def test_reviewmsg_reply_reports_configured_action_when_policy_acts() -> None:
    """With an acting policy the reply must state the submitted action -- and with
    a non-acting config the reply must never claim the author was "actioned".
    The old unconditional reviewmsg_result told moderators the message was
    handled even on servers whose policy meant the bot deliberately did
    nothing (the default!), sending them hunting for a delivery bug that was
    actually a config setting."""
    deps = FakeDeps(config={"action_policy": "delete_ban", "review_channel": 444})
    resp = await handle_command(
        _review_ctx(attachments=[(1, "https://x/1.png")], author_id=333), deps
    )
    assert resp.i18n_key == "command.reviewmsg_result_submitted"
    assert resp.params["action"] == "delete_ban"
    assert resp.params["author_id"] == 333
    message = render(resp, "en")
    assert "Submitted delete_ban" in message
    assert "Queued" not in message
    assert "<#444>" in message
    assert "final outcome" in message
    # The verdict still reaches the pipeline regardless of the reply wording.
    assert len(deps.confirmed_scams) == 1


@pytest.mark.asyncio
async def test_reviewmsg_reply_explains_async_outcome_without_review_channel() -> None:
    deps = FakeDeps(config={"action_policy": "delete_ban"})
    resp = await handle_command(_review_ctx(), deps)

    assert resp.i18n_key == "command.reviewmsg_result_submitted_no_channel"
    message = render(resp, "en")
    assert "Submitted delete_ban" in message
    assert "asynchronously" in message
    assert "does not show its final outcome" in message


@pytest.mark.asyncio
async def test_reviewmsg_reply_says_report_only_under_report_only_policy() -> None:
    deps = FakeDeps(config={"action_policy": "report_only"})
    resp = await handle_command(_review_ctx(attachments=[(1, "https://x/1.png")]), deps)
    assert resp.i18n_key == "command.reviewmsg_result_report_only"
    assert resp.params["action"] == "report_only"
    assert len(deps.confirmed_scams) == 1


@pytest.mark.asyncio
async def test_reviewmsg_reply_says_safe_mode_even_with_acting_policy() -> None:
    """Safe mode overrides the configured action in policy.decide, so the
    reply must lead with safe mode, not the (suppressed) acting policy."""
    deps = FakeDeps(config={"action_policy": "delete_ban", "safe_mode": True})
    resp = await handle_command(_review_ctx(attachments=[(1, "https://x/1.png")]), deps)
    assert resp.i18n_key == "command.reviewmsg_result_safe_mode"
    assert len(deps.confirmed_scams) == 1


@pytest.mark.asyncio
async def test_context_menu_review_message_routes_to_shared_core() -> None:
    """The context-menu entry point (``review_message``) shares ``_review_message``
    with the slash command -- both are gated by the same permission and both
    read the glue-resolved options in the same shape."""
    deps = FakeDeps()
    ctx = _review_ctx(command="review_message", subcommand=None)
    resp = await handle_command(ctx, deps)
    assert resp.i18n_key == "command.reviewmsg_result_report_only"
    assert len(deps.confirmed_scams) == 1


@pytest.mark.asyncio
async def test_context_menu_review_message_denied_without_manage_guild() -> None:
    ctx = _review_ctx(command="review_message", subcommand=None, perms=NONE)
    with pytest.raises(InteractionRejected) as exc:
        await handle_command(ctx, FakeDeps())
    assert exc.value.reason is CommandError.NO_PERMISSION
