"""Server-side interaction handlers: auth, side effects, audit — hikari-free.

Each slash command and component (button) press is reduced to a plain
:class:`InteractionContext` (who, where, which command, which options) and
dispatched here. Handlers run the *server-side* permission re-check, perform the
database/Redis side effects through injected dependencies, write a
``mod_actions`` audit row for every state change, and return an
:class:`InteractionResponse` (always ephemeral) carrying an i18n key.

Nothing in this module imports hikari, so the permission matrix, audit
behaviour, and appeal lifecycle are fully unit-testable. The hikari/REST/DB
wiring that produces an :class:`InteractionContext` and renders an
:class:`InteractionResponse` lives in :mod:`.service`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from optimus.db.models import GuildHash, GuildWhitelist
from optimus.globaldb.service import GlobalHashService, SubmissionDenied
from optimus.services.interactions.commands import required_permission
from optimus.services.interactions.logic import (
    CommandError,
    ComponentAction,
    InteractionRejected,
    Permission,
    build_export,
    has_permission,
    parse_hash_hex,
    validate_config_set,
    validate_import,
)
from optimus.services.interactions.logic import (
    ImportHash as _ImportHash,
)
from optimus.services.moderation.review import ParsedCustomId, ReviewAction


@dataclass(frozen=True, slots=True)
class InteractionContext:
    """Everything a handler needs about one invocation, gateway-agnostic."""

    guild_id: int | None
    user_id: int
    #: The invoking member's *effective* permission bitfield (never the hint).
    member_permissions: int
    command: str
    subcommand: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    locale: str = "en"


@dataclass(frozen=True, slots=True)
class InteractionResponse:
    """An always-ephemeral reply, identified by an i18n key plus params."""

    i18n_key: str
    params: dict[str, Any] = field(default_factory=dict)
    #: Optional opaque payload (e.g. an export file body) for the glue layer.
    attachment: str | None = None


class InteractionDeps(Protocol):
    """Side-effecting collaborators a handler needs, all per-request scoped."""

    async def add_guild_hash(self, guild_id: int, gh: GuildHash) -> GuildHash: ...
    async def remove_guild_hash(self, guild_id: int, hash_id: str) -> int: ...
    async def list_guild_hashes(self, guild_id: int) -> list[GuildHash]: ...
    async def add_whitelist(self, guild_id: int, entry: GuildWhitelist) -> GuildWhitelist: ...
    async def get_config(self, guild_id: int) -> dict[str, Any]: ...
    async def set_config_field(self, guild_id: int, field: str, value: Any) -> None: ...
    async def stats_summary(self, guild_id: int) -> dict[str, Any]: ...
    async def opt_out_user(self, user_id: int) -> int: ...
    async def purge_guild(self, guild_id: int) -> int: ...
    async def recent_detection_for(self, guild_id: int, user_id: int) -> int | None: ...
    async def open_appeal(self, guild_id: int, detection_id: int, user_id: int) -> int: ...
    async def get_appeal(self, guild_id: int, appeal_id: int) -> dict[str, Any] | None: ...
    async def resolve_appeal(self, guild_id: int, appeal_id: int, *, approved: bool) -> None: ...
    async def reverse_detection_action(self, guild_id: int, detection_id: int) -> None: ...
    async def disable_safe_mode(self, guild_id: int) -> None: ...
    async def local_hash(self, guild_id: int, hash_id: str) -> GuildHash | None: ...
    async def hash_rate_ok(self, user_id: int) -> bool: ...
    async def appeal_cooldown_ok(self, user_id: int) -> bool: ...
    async def audit(
        self, guild_id: int, actor_id: int, action: str, *, target: str | None = None
    ) -> None: ...
    def global_service(self) -> GlobalHashService: ...


def _require(ctx: InteractionContext, permission: Permission | None) -> None:
    """Enforce guild-only + server-side permission, raising on failure."""
    if permission is not None and ctx.guild_id is None:
        raise InteractionRejected(CommandError.GUILD_ONLY)
    if permission is not None and not has_permission(ctx.member_permissions, permission):
        raise InteractionRejected(CommandError.NO_PERMISSION)


async def handle_command(ctx: InteractionContext, deps: InteractionDeps) -> InteractionResponse:
    """Dispatch a slash command to its handler after the auth gate."""
    _require(ctx, required_permission(ctx.command))
    handler = _COMMAND_HANDLERS.get(ctx.command)
    if handler is None:  # pragma: no cover - registration guarantees coverage
        raise InteractionRejected(CommandError.UNKNOWN_FIELD)
    return await handler(ctx, deps)


async def _cmd_scamhash(ctx: InteractionContext, deps: InteractionDeps) -> InteractionResponse:
    assert ctx.guild_id is not None  # guaranteed by _require (MANAGE_GUILD => guild-only)
    sub = ctx.subcommand
    if sub == "add":
        if not await deps.hash_rate_ok(ctx.user_id):
            raise InteractionRejected(CommandError.RATE_LIMITED)
        gh = _build_hash_from_options(ctx.options, added_by=ctx.user_id)
        stored = await deps.add_guild_hash(ctx.guild_id, gh)
        await deps.audit(ctx.guild_id, ctx.user_id, "scamhash.add", target=stored.hash_id)
        return InteractionResponse("command.hash_added", {"hash_id": stored.hash_id})
    if sub == "remove":
        hash_id = str(ctx.options["hash_id"])
        removed = await deps.remove_guild_hash(ctx.guild_id, hash_id)
        if removed == 0:
            return InteractionResponse("command.hash_not_found", {"hash_id": hash_id})
        await deps.audit(ctx.guild_id, ctx.user_id, "scamhash.remove", target=hash_id)
        return InteractionResponse("command.hash_removed", {"hash_id": hash_id})
    if sub == "list":
        rows = await deps.list_guild_hashes(ctx.guild_id)
        if not rows:
            return InteractionResponse("command.hash_list_empty")
        return InteractionResponse("command.hash_list_header", {"count": len(rows)})
    if sub == "import":
        entries = validate_import(str(ctx.options["file"]))
        added = await _import_hashes(deps, ctx.guild_id, entries, added_by=ctx.user_id)
        await deps.audit(ctx.guild_id, ctx.user_id, "scamhash.import", target=str(added))
        return InteractionResponse(
            "command.import_ok", {"added": added, "skipped": len(entries) - added}
        )
    if sub == "export":
        rows = await deps.list_guild_hashes(ctx.guild_id)
        body = build_export(
            [_ImportHash(phash=r.phash, dhash=r.dhash, whash=r.whash) for r in rows]
        )
        return InteractionResponse("command.export_ok", {"count": len(rows)}, attachment=body)
    raise InteractionRejected(CommandError.UNKNOWN_FIELD)  # pragma: no cover


async def _cmd_config(ctx: InteractionContext, deps: InteractionDeps) -> InteractionResponse:
    assert ctx.guild_id is not None
    if ctx.subcommand == "view":
        await deps.get_config(ctx.guild_id)
        return InteractionResponse("command.config_view_header")
    change = validate_config_set(str(ctx.options["field"]), str(ctx.options["value"]))
    await deps.set_config_field(ctx.guild_id, change.field, change.value)
    await deps.audit(ctx.guild_id, ctx.user_id, "config.set", target=change.field)
    return InteractionResponse(
        "command.config_set_ok", {"field": change.field, "value": change.value}
    )


async def _cmd_stats(ctx: InteractionContext, deps: InteractionDeps) -> InteractionResponse:
    assert ctx.guild_id is not None
    summary = await deps.stats_summary(ctx.guild_id)
    if not summary or summary.get("detections", 0) == 0:
        return InteractionResponse("command.stats_empty")
    return InteractionResponse("command.stats_header", {"hours": summary.get("hours", 24)})


async def _cmd_submit_global(ctx: InteractionContext, deps: InteractionDeps) -> InteractionResponse:
    assert ctx.guild_id is not None
    hash_id = str(ctx.options["hash_id"])
    local = await deps.local_hash(ctx.guild_id, hash_id)
    if local is None:
        return InteractionResponse("command.hash_not_found", {"hash_id": hash_id})
    try:
        await deps.global_service().submit(
            hash_id=local.hash_id,
            phash=local.phash,
            dhash=local.dhash,
            whash=local.whash,
            submitter_user_id=ctx.user_id,
            submitter_guild_id=ctx.guild_id,
        )
    except SubmissionDenied as denied:
        if denied.reason == "rate_limited":
            raise InteractionRejected(CommandError.RATE_LIMITED) from denied
        raise InteractionRejected(CommandError.BELOW_THRESHOLD) from denied
    await deps.audit(ctx.guild_id, ctx.user_id, "global.submit", target=hash_id)
    return InteractionResponse("command.submit_global_ok", {"hash_id": hash_id})


async def _cmd_delete_server_data(
    ctx: InteractionContext, deps: InteractionDeps
) -> InteractionResponse:
    # The destructive purge itself is gated behind the confirm button
    # (component handler); this only renders the confirmation prompt.
    return InteractionResponse("command.delete_server_confirm")


async def _cmd_forget_me(ctx: InteractionContext, deps: InteractionDeps) -> InteractionResponse:
    await deps.opt_out_user(ctx.user_id)
    if ctx.guild_id is not None:
        await deps.audit(ctx.guild_id, ctx.user_id, "forget_me", target=str(ctx.user_id))
    return InteractionResponse("command.forget_me_ok")


async def _cmd_appeal(ctx: InteractionContext, deps: InteractionDeps) -> InteractionResponse:
    assert ctx.guild_id is not None
    if not await deps.appeal_cooldown_ok(ctx.user_id):
        return InteractionResponse("dm.appeal_cooldown")
    detection_id = await deps.recent_detection_for(ctx.guild_id, ctx.user_id)
    if detection_id is None:
        return InteractionResponse("command.appeal_none")
    await deps.open_appeal(ctx.guild_id, detection_id, ctx.user_id)
    await deps.audit(ctx.guild_id, ctx.user_id, "appeal.open", target=str(detection_id))
    return InteractionResponse("command.appeal_opened")


_CommandHandler = Callable[
    ["InteractionContext", "InteractionDeps"], Awaitable["InteractionResponse"]
]

_COMMAND_HANDLERS: dict[str, _CommandHandler] = {
    "scamhash": _cmd_scamhash,
    "config": _cmd_config,
    "stats": _cmd_stats,
    "submit_global": _cmd_submit_global,
    "delete_server_data": _cmd_delete_server_data,
    "forget_me": _cmd_forget_me,
    "appeal": _cmd_appeal,
}


def _build_hash_from_options(options: dict[str, Any], *, added_by: int) -> GuildHash:
    """Build a :class:`GuildHash` from ``/scamhash add`` options.

    Accepts either a precomputed image hash triple supplied by the glue layer
    (``phash``/``dhash``/``whash`` as ints, e.g. hashed from an attachment) or
    hex strings the user typed. ``hash_id`` is derived deterministically.
    """
    p, d, w = options.get("phash"), options.get("dhash"), options.get("whash")
    if isinstance(p, int) and isinstance(d, int) and isinstance(w, int):
        phash, dhash, whash = p, d, w
    else:
        phash = parse_hash_hex(str(p))
        dhash = parse_hash_hex(str(d)) if d is not None else 0
        whash = parse_hash_hex(str(w)) if w is not None else 0
    hash_id = options.get("hash_id") or f"{phash:016x}"
    return GuildHash(
        hash_id=str(hash_id),
        phash=phash,
        dhash=dhash,
        whash=whash,
        ahash=0,
        source="local",
        added_by=added_by,
    )


async def _import_hashes(
    deps: InteractionDeps, guild_id: int, entries: list[_ImportHash], *, added_by: int
) -> int:
    added = 0
    seen: set[str] = set()
    for entry in entries:
        hash_id = f"{entry.phash:016x}"
        if hash_id in seen:
            continue
        seen.add(hash_id)
        await deps.add_guild_hash(
            guild_id,
            GuildHash(
                hash_id=hash_id,
                phash=entry.phash,
                dhash=entry.dhash,
                whash=entry.whash,
                ahash=0,
                source="import",
                added_by=added_by,
            ),
        )
        added += 1
    return added


# --- component (button) handlers -------------------------------------------------


async def handle_review_button(
    ctx: InteractionContext, parsed: ParsedCustomId, deps: InteractionDeps
) -> InteractionResponse:
    """Handle a report button after re-checking the clicker's permission.

    Every report action is a state change requiring ``MANAGE_GUILD``; the check
    runs on *this* click's member permissions, never the message's original
    author or any cached value.
    """
    _require(ctx, Permission.MANAGE_GUILD)
    assert ctx.guild_id is not None
    action = parsed.action
    detection_id = parsed.detection_id

    if action is ReviewAction.CONFIRM_SCAM:
        await deps.audit(ctx.guild_id, ctx.user_id, "review.confirm_scam", target=str(detection_id))
        return InteractionResponse("button.confirmed_scam", {"detection_id": detection_id})
    if action is ReviewAction.FALSE_POSITIVE:
        await deps.reverse_detection_action(ctx.guild_id, detection_id)
        await deps.audit(
            ctx.guild_id, ctx.user_id, "review.false_positive", target=str(detection_id)
        )
        return InteractionResponse("button.marked_false_positive", {"detection_id": detection_id})
    if action is ReviewAction.BAN_UPLOADER:
        await deps.audit(ctx.guild_id, ctx.user_id, "review.ban_uploader", target=str(detection_id))
        return InteractionResponse("button.uploader_banned")
    if action is ReviewAction.UNBAN:
        await deps.audit(ctx.guild_id, ctx.user_id, "review.unban", target=str(detection_id))
        return InteractionResponse("button.uploader_unbanned")
    if action is ReviewAction.WHITELIST_IMAGE:
        await deps.audit(
            ctx.guild_id, ctx.user_id, "review.whitelist_image", target=str(detection_id)
        )
        return InteractionResponse("button.image_whitelisted")
    if action is ReviewAction.SUBMIT_GLOBAL:
        await deps.audit(
            ctx.guild_id, ctx.user_id, "review.submit_global", target=str(detection_id)
        )
        return InteractionResponse("button.submitted_global")
    raise InteractionRejected(CommandError.UNKNOWN_FIELD)  # pragma: no cover


async def handle_component(
    ctx: InteractionContext, action: ComponentAction, ref_id: int, deps: InteractionDeps
) -> InteractionResponse:
    """Handle a non-report component (appeal lifecycle, safe-mode, purge confirm)."""
    if action is ComponentAction.APPEAL_OPEN:
        if not await deps.appeal_cooldown_ok(ctx.user_id):
            return InteractionResponse("dm.appeal_cooldown")
        assert ctx.guild_id is not None
        await deps.open_appeal(ctx.guild_id, ref_id, ctx.user_id)
        await deps.audit(ctx.guild_id, ctx.user_id, "appeal.open", target=str(ref_id))
        return InteractionResponse("dm.appeal_submitted")

    # The remaining controls are all moderator/admin state changes.
    if action in (ComponentAction.APPEAL_APPROVE, ComponentAction.APPEAL_DENY):
        _require(ctx, Permission.MANAGE_GUILD)
        assert ctx.guild_id is not None
        approved = action is ComponentAction.APPEAL_APPROVE
        await deps.resolve_appeal(ctx.guild_id, ref_id, approved=approved)
        if approved:
            appeal = await deps.get_appeal(ctx.guild_id, ref_id)
            if appeal is not None:
                detection_id = int(appeal["detection_id"])
                await deps.reverse_detection_action(ctx.guild_id, detection_id)
            await deps.audit(ctx.guild_id, ctx.user_id, "appeal.approve", target=str(ref_id))
            return InteractionResponse("button.appeal_approved")
        await deps.audit(ctx.guild_id, ctx.user_id, "appeal.deny", target=str(ref_id))
        return InteractionResponse("button.appeal_denied")

    if action is ComponentAction.SAFE_MODE_RESUME:
        _require(ctx, Permission.MANAGE_GUILD)
        assert ctx.guild_id is not None
        await deps.disable_safe_mode(ctx.guild_id)
        await deps.audit(ctx.guild_id, ctx.user_id, "safe_mode.resume")
        return InteractionResponse("button.safe_mode_resumed")

    if action is ComponentAction.DELETE_SERVER_CONFIRM:
        _require(ctx, Permission.ADMINISTRATOR)
        assert ctx.guild_id is not None
        # A full GDPR purge erases the audit log too, so recording a row here
        # would be immediately deleted; the purge is the audited event itself.
        await deps.purge_guild(ctx.guild_id)
        return InteractionResponse("command.delete_server_ok")

    raise InteractionRejected(CommandError.UNKNOWN_FIELD)  # pragma: no cover
