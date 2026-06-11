"""Interaction handlers: server-side auth, audit, button auth, appeal lifecycle."""

from __future__ import annotations

from typing import Any

import pytest

from optimus.db.models import GuildHash, GuildWhitelist
from optimus.globaldb.service import GlobalHashService, SubmissionDenied
from optimus.services.interactions.handlers import (
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
        self._next_appeal_id = 1
        self._global_service = _FakeGlobalService(flags.get("submit_error"))
        self.global_submitted: list[str] = []

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
        return {"locale": "en"}

    async def set_config_field(self, guild_id: int, field: str, value: Any) -> None:
        self.config_set.append((field, value))

    async def stats_summary(self, guild_id: int) -> dict[str, Any]:
        return {"detections": 3, "hours": 24}

    async def opt_out_user(self, user_id: int) -> int:
        self.opted_out.append(user_id)
        return 1

    async def purge_guild(self, guild_id: int) -> int:
        self.purged.append(guild_id)
        return 7

    async def recent_detection_for(self, guild_id: int, user_id: int) -> int | None:
        return self._recent_detection

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
    resp = await handle_command(
        _ctx("scamhash", subcommand="add", phash="deadbeef"), deps
    )
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
    deps.hashes["abc"] = GuildHash(
        hash_id="abc", phash=1, dhash=2, whash=3, ahash=0, source="local"
    )
    resp = await handle_command(_ctx("scamhash", subcommand="list"), deps)
    assert resp.i18n_key == "command.hash_list_header"
    assert resp.params["count"] == 1


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


@pytest.mark.asyncio
async def test_stats_non_empty() -> None:
    resp = await handle_command(_ctx("stats"), FakeDeps())
    assert resp.i18n_key == "command.stats_header"
    assert resp.params["hours"] == 24


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
