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

from optimus.core.logging import get_logger
from optimus.db.models import GuildHash, GuildWhitelist
from optimus.globaldb.service import GlobalHashService, SubmissionDenied
from optimus.services.interactions.attachment_hash import (
    AttachmentHashError,
    AttachmentHashes,
)
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

_log = get_logger(__name__)

#: Keep ``/scamhash list`` safely below Discord's 2,000-character message limit
#: even if every rendered line maxes out (64-char hash id, 32-char source, and
#: a full-width ``by <@user>`` mention ≈ 130 chars per line).
_HASH_LIST_PREVIEW_LIMIT = 12


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
    async def detection_belongs_to(
        self, guild_id: int, detection_id: int, user_id: int
    ) -> bool: ...
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
    async def compute_attachment_hashes(self, *, attachment_id: int, url: str) -> AttachmentHashes:
        """Fetch and decode one attachment and compute its hash set. No DB access.

        Deliberately split out from storing the result: this does a network
        fetch plus a sandboxed decode subprocess, both of which can take real
        wall-clock time (multi-second on a loaded host) and must never run
        while a DB write transaction is open -- SQLite holds an exclusive
        file-level write lock for the full lifetime of the transaction, and a
        review of a multi-image message previously ran every attachment's
        fetch+decode one after another *inside* the same open transaction as
        the DB writes, which could hold that lock far longer than any normal
        query and starve concurrent writers into a "database is locked" error.

        Raises :class:`AttachmentHashError` (see
        :mod:`optimus.services.interactions.attachment_hash`) if the attachment
        cannot be fetched or decoded as an image; the caller decides how to
        surface that (skip-and-continue for a multi-image review).
        """
        ...

    async def store_attachment_hash(
        self, guild_id: int, *, hashes: AttachmentHashes, added_by: int
    ) -> GuildHash:
        """Store an already-computed attachment hash set as a guild hash.

        DB-only -- no network or decode work happens here, so this is always
        fast and holds the session's write lock only as long as one insert
        takes. If a hash with the same id already exists for this guild (e.g.
        re-reviewing a message, or an image an earlier detection already
        caught), returns the existing row rather than raising -- adding a scam
        hash is idempotent.
        """
        ...

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
        """Record a moderator-confirmed scam match and run the moderation pipeline.

        Feeds the same ``verdict.v1`` path a live detection would, so the
        guild's configured ``action_policy`` (e.g. delete + ban) is applied
        exactly as it would be for a message caught in real time.
        """
        ...


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
        shown = sorted(rows, key=lambda r: r.hash_id)[:_HASH_LIST_PREVIEW_LIMIT]
        params: dict[str, Any] = {
            "count": len(rows),
            "hashes": "\n".join(_render_hash_entry(r) for r in shown),
        }
        if len(rows) <= _HASH_LIST_PREVIEW_LIMIT:
            return InteractionResponse("command.hash_list_header", params)
        params["remaining"] = len(rows) - len(shown)
        return InteractionResponse("command.hash_list_truncated", params)
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
    if sub == "reviewmsg":
        return await _review_message(ctx, deps)
    raise InteractionRejected(CommandError.UNKNOWN_FIELD)  # pragma: no cover


def _render_hash_entry(row: GuildHash) -> str:
    """One display line per hash: id, source, and who added it (when known)."""
    added_by = f" by <@{row.added_by}>" if row.added_by is not None else ""
    return f"\u2022 `{row.hash_id}` \u2014 {row.source}{added_by}"


async def _cmd_review_message(
    ctx: InteractionContext, deps: InteractionDeps
) -> InteractionResponse:
    """Entry point for the "Review as scam" message context-menu command.

    ``required_permission("review_message")`` gates this the same as
    ``/scamhash reviewmsg`` (``MANAGE_GUILD``); the glue layer has already
    resolved the target message's attachments/author into ``ctx.options``
    since a context-menu command carries no typed options of its own.
    """
    return await _review_message(ctx, deps)


async def _review_message(ctx: InteractionContext, deps: InteractionDeps) -> InteractionResponse:
    """Shared core for both the ``/scamhash reviewmsg`` and context-menu entry points.

    Expects the glue layer to have pre-resolved the target message into
    ``ctx.options``: ``channel_id``, ``message_id``, ``author_id`` (all ints),
    and ``attachments`` (a list of ``(attachment_id, url)`` pairs already
    filtered to image content types). Hashes every attachment, adds each as a
    new guild hash, and -- for each one successfully hashed -- feeds a
    confirmed-scam verdict into the moderation pipeline so the configured
    action policy (e.g. delete + ban) is applied. A REST-level failure to
    fetch/decode one attachment is skipped rather than aborting the whole
    review, since a message can carry several images and a moderator's intent
    is best served by processing every image that *can* be processed.

    Runs in two passes deliberately: first every attachment is fetched and
    hashed with no DB session/transaction involved at all, then (only once
    all the slow network/decode work is done) each successfully hashed
    attachment is stored and submitted in quick DB-only calls. The whole
    handler still executes inside one caller-managed transaction (see
    :meth:`InteractionService._run`), so interleaving a network fetch plus a
    sandboxed decode subprocess for attachment N+1 with attachment N's writes
    used to hold that transaction's SQLite write lock open for as long as an
    entire multi-image review took -- multiple seconds per image, easily
    exceeding the point another writer would report "database is locked".
    Doing all the slow work up front means the transaction's write lock is
    only ever held for the sum of the fast DB calls, not the fetch+decode
    time too.
    """
    assert ctx.guild_id is not None  # guaranteed by _require (MANAGE_GUILD => guild-only)
    if not await deps.hash_rate_ok(ctx.user_id):
        raise InteractionRejected(CommandError.RATE_LIMITED)
    channel_id = int(ctx.options["channel_id"])
    message_id = int(ctx.options["message_id"])
    author_id = int(ctx.options["author_id"])
    attachments: list[tuple[int, str]] = list(ctx.options["attachments"])
    if not attachments:
        return InteractionResponse("command.reviewmsg_no_images")

    # Pass 1: fetch + decode + hash every attachment. Pure computation plus
    # network I/O -- deliberately kept outside any DB write below so the
    # transaction the caller already has open never sits idle waiting on a
    # CDN round-trip or a sandboxed decode subprocess.
    computed: list[tuple[int, AttachmentHashes]] = []
    failed = 0
    for attachment_id, url in attachments:
        try:
            hashes = await deps.compute_attachment_hashes(attachment_id=attachment_id, url=url)
        except AttachmentHashError as exc:
            _log.warning(
                "reviewmsg_attachment_hash_failed",
                guild_id=ctx.guild_id,
                attachment_id=attachment_id,
                reason=str(exc),
            )
            failed += 1
            continue
        computed.append((attachment_id, hashes))

    # Pass 2: store + audit + submit. DB-only, no network/decode work, so
    # each iteration is fast and the write lock is held for close to the
    # minimum time actually needed.
    added_hash_ids: list[str] = []
    for attachment_id, hashes in computed:
        stored = await deps.store_attachment_hash(ctx.guild_id, hashes=hashes, added_by=ctx.user_id)
        added_hash_ids.append(stored.hash_id)
        await deps.audit(ctx.guild_id, ctx.user_id, "scamhash.reviewmsg", target=stored.hash_id)
        await deps.submit_confirmed_scam(
            ctx.guild_id,
            channel_id=channel_id,
            message_id=message_id,
            attachment_id=attachment_id,
            uploader_id=author_id,
            matched_hash_id=stored.hash_id,
        )
    if not added_hash_ids:
        return InteractionResponse("command.reviewmsg_all_failed", {"failed": failed})

    # Report what the moderation pipeline will actually do with the submitted
    # verdicts, mirroring optimus.services.moderation.policy.decide for a
    # confirmed verdict (SCAM, confidence 1.0 -- always clears the auto-act
    # bar): safe mode and a report_only/none policy both mean "report only,
    # nothing deleted". The old unconditional "actioned <@user>" reply told a
    # moderator the message was handled even when the configured policy meant
    # the bot would deliberately do nothing to it.
    config = await deps.get_config(ctx.guild_id)
    policy = str(config.get("action_policy") or "report_only")
    params: dict[str, Any] = {
        "added": len(added_hash_ids),
        "failed": failed,
        "author_id": author_id,
        "action": policy,
    }
    if bool(config.get("safe_mode", False)):
        return InteractionResponse("command.reviewmsg_result_safe_mode", params)
    if policy in ("none", "report_only"):
        return InteractionResponse("command.reviewmsg_result_report_only", params)
    review_channel = config.get("review_channel")
    if review_channel is None:
        return InteractionResponse("command.reviewmsg_result_submitted_no_channel", params)
    params["review_channel"] = review_channel
    return InteractionResponse("command.reviewmsg_result_submitted", params)


async def _cmd_config(ctx: InteractionContext, deps: InteractionDeps) -> InteractionResponse:
    assert ctx.guild_id is not None
    if ctx.subcommand == "view":
        current = await deps.get_config(ctx.guild_id)
        return InteractionResponse(
            "command.config_view_header", {"summary": _render_config_summary(current)}
        )
    change = validate_config_set(str(ctx.options["field"]), str(ctx.options["value"]))
    await deps.set_config_field(ctx.guild_id, change.field, change.value)
    await deps.audit(ctx.guild_id, ctx.user_id, "config.set", target=change.field)
    return InteractionResponse(
        "command.config_set_ok",
        {"field": change.field, "value": _render_config_value(change.field, change.value)},
    )


#: Display order for /config view; keeps related settings grouped together.
#: These keys must exactly match both the dict keys returned by
#: InteractionDeps.get_config() and the field names validate_config_set()
#: accepts for /config set -- i.e. "review_channel", never the DB column's
#: "review_channel_id" -- so a field name copied from /config view always
#: works verbatim in /config set and vice versa.
_CONFIG_VIEW_ORDER = (
    "sensitivity",
    "action_policy",
    "mod_queue_threshold",
    "review_channel",
    "ban_purge_hours",
    "safe_mode",
    "retention_days",
    "locale",
    "optin_global_db",
    "optin_scan_bots",
    "optin_evidence_storage",
)


def _render_config_summary(current: dict[str, Any]) -> str:
    """Render a guild's config dict (from ``get_config``) as a display block.

    Empty (no row yet / guild never configured) renders a single explanatory
    line rather than an empty list. ``review_channel`` renders as a real
    channel mention (or "not set") to match ``_render_config_value``.
    """
    if not current:
        return "_No configuration set yet \u2014 defaults are in effect._"
    lines = []
    for config_field in _CONFIG_VIEW_ORDER:
        if config_field not in current:
            continue
        value = current[config_field]
        if config_field == "review_channel":
            rendered = f"<#{value}>" if value is not None else "not set"
        else:
            rendered = str(value)
        lines.append(f"**{config_field}**: `{rendered}`")
    return "\n".join(lines)


def _render_config_value(field: str, value: Any) -> str:
    """Render a validated config value for the ``config_set_ok`` confirmation.

    ``review_channel`` stores a raw channel id (or ``None`` when cleared); show
    it as a real channel mention (or "none") instead of a bare integer/"None".
    Every other field renders as-is.
    """
    if field == "review_channel":
        return f"<#{value}>" if value is not None else "none"
    return str(value)


async def _cmd_stats(ctx: InteractionContext, deps: InteractionDeps) -> InteractionResponse:
    assert ctx.guild_id is not None
    summary = await deps.stats_summary(ctx.guild_id)
    if not summary:
        return InteractionResponse("command.stats_empty")
    # Zero detections still renders the header: the database line doubles as the
    # persistence canary (boot count + stable first-boot date), which moderators
    # need to see on quiet servers too.
    return InteractionResponse(
        "command.stats_header",
        {
            "hours": summary.get("hours", 24),
            "detections": summary.get("detections", 0),
            "boots": summary.get("boots", 0),
            "first_boot": summary.get("first_boot", "unknown"),
        },
    )


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
    "review_message": _cmd_review_message,
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
    # Mirror (flip) hashes are supplied as ints by the glue layer only when it
    # hashed an actual attachment (it also flips the pixels); typed-hex adds have
    # no image and leave them NULL.
    mp, md, mw = options.get("mphash"), options.get("mdhash"), options.get("mwhash")
    mirror_given = isinstance(mp, int) and isinstance(md, int) and isinstance(mw, int)
    return GuildHash(
        hash_id=str(hash_id),
        phash=phash,
        dhash=dhash,
        whash=whash,
        ahash=0,
        mphash=mp if mirror_given else None,
        mdhash=md if mirror_given else None,
        mwhash=mw if mirror_given else None,
        mahash=options.get("mahash") if mirror_given else None,
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
        if ctx.guild_id is None:
            raise InteractionRejected(CommandError.GUILD_ONLY)
        # The detection id rides in the (client-echoed, forgeable) custom id, so
        # never trust it: only the user the detection was filed against may appeal
        # it, and only within the detection's own guild. The /appeal command path
        # derives this server-side; here we re-verify ownership explicitly.
        if not await deps.detection_belongs_to(ctx.guild_id, ref_id, ctx.user_id):
            return InteractionResponse("command.appeal_none")
        if not await deps.appeal_cooldown_ok(ctx.user_id):
            return InteractionResponse("dm.appeal_cooldown")
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
