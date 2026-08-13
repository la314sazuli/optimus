"""hikari runtime for slash commands and component interactions.

This is the thin, side-effecting shell around the pure handlers in
:mod:`.handlers`. It:

* turns a hikari interaction into a gateway-agnostic
  :class:`~optimus.services.interactions.handlers.InteractionContext` (resolving
  the invoker's *effective* permissions server-side);
* implements :class:`~optimus.services.interactions.handlers.InteractionDeps`
  against per-request database sessions, Redis, and the global hash service;
* renders every handler result as an **ephemeral** response, mapping the
  machine-readable rejection reason to a localized i18n string.

All database work runs inside a fresh :func:`session_scope` per interaction so a
handler failure rolls back cleanly and never leaks a half-applied state change.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

from sqlalchemy.exc import OperationalError

from optimus.contracts.events import Verdict, VerdictEvent
from optimus.core.backoff import BackoffPolicy, retry_async
from optimus.core.config import Settings
from optimus.core.logging import correlation_context, get_correlation_id, get_logger
from optimus.core.ratelimit import RateLimit, RateLimiter
from optimus.db.engine import SessionScope
from optimus.db.models import Guild, GuildHash, GuildWhitelist
from optimus.db.repositories import (
    AppealRepository,
    DetectionRepository,
    GlobalHashRepository,
    GlobalSubmitterRepository,
    GuildHashRepository,
    GuildPurgeRepository,
    GuildRepository,
    ModActionRepository,
    UserOptoutRepository,
    WhitelistRepository,
)
from optimus.globaldb.service import GlobalHashService
from optimus.i18n import translate
from optimus.ingest.fetcher import FetchedImage, fetch_image
from optimus.services.interactions.attachment_hash import (
    AttachmentHashes,
    FetchFn,
    hash_attachment,
)
from optimus.services.interactions.handlers import (
    InteractionContext,
    InteractionResponse,
    handle_command,
    handle_component,
    handle_review_button,
)
from optimus.services.interactions.logic import (
    CommandError,
    InteractionRejected,
    decode_component_id,
)
from optimus.services.moderation.review import decode_custom_id

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from optimus.services.detection.service import DetectionService

_log = get_logger(__name__)


def _default_fetch(settings: Settings) -> FetchFn:
    """Build the same bounded fetch used by the passive ingest pipeline.

    Reusing ``ingest_max_bytes``/``ingest_max_redirects`` keeps a
    command-driven hash (``/scamhash add image:``, message review) under the
    same SSRF/size guarantees as an attachment the bot observes live.
    """
    fetch = partial(
        fetch_image,
        max_bytes=settings.ingest_max_bytes,
        max_redirects=settings.ingest_max_redirects,
    )

    async def _fetch(url: str) -> FetchedImage:
        return await fetch(url)

    return _fetch


#: Per-user budgets for the rate-limited commands.
HASH_RATE = RateLimit(capacity=10.0, refill_rate=1.0 / 6.0)

#: Maps a rejection reason to the i18n key for its ephemeral message. Reasons
#: whose enum value differs from the catalog suffix are remapped explicitly.
_ERROR_KEYS: dict[CommandError, str] = {
    CommandError.NO_PERMISSION: "command.no_permission",
    CommandError.GUILD_ONLY: "command.guild_only",
    CommandError.RATE_LIMITED: "command.rate_limited",
    CommandError.INVALID_HEX: "command.hash_invalid_hex",
    CommandError.IMPORT_INVALID: "command.import_invalid",
    CommandError.IMPORT_TOO_LARGE: "command.import_too_large",
    CommandError.UNKNOWN_FIELD: "command.config_unknown_field",
    CommandError.INVALID_VALUE: "command.config_invalid_value",
    CommandError.BELOW_THRESHOLD: "command.submit_global_below_threshold",
    CommandError.MESSAGE_NOT_FOUND: "command.reviewmsg_not_found",
    CommandError.FETCH_FAILED: "command.reviewmsg_fetch_failed",
}


def error_message(reason: CommandError, locale: str) -> str:
    """Localize a rejection reason for display to the invoker."""
    key = _ERROR_KEYS[reason]
    params: dict[str, Any] = {}
    if reason in (CommandError.IMPORT_INVALID, CommandError.INVALID_VALUE):
        params["reason"] = reason.value
    if reason is CommandError.UNKNOWN_FIELD:
        params["field"] = "?"
    if reason is CommandError.IMPORT_TOO_LARGE:
        params["limit"] = 1000
    return translate(key, locale, **params)


class DbDeps:
    """:class:`InteractionDeps` over a single session, Redis, and the global service."""

    def __init__(
        self,
        session: AsyncSession,
        rate_limiter: RateLimiter,
        settings: Settings,
        *,
        appeal_cooldown_seconds: int = 3600,
        fetch: FetchFn | None = None,
        detection: DetectionService | None = None,
    ) -> None:
        self._session = session
        self._rl = rate_limiter
        self._settings = settings
        self._appeal_cooldown = appeal_cooldown_seconds
        self._fetch = fetch or _default_fetch(settings)
        self._detection = detection
        #: Confirmed-scam verdicts persisted in this request's transaction but
        #: not yet published to the moderation bus. :meth:`InteractionService._run`
        #: publishes these only after the transaction commits -- publishing
        #: earlier would let the moderation consumer (which opens its own DB
        #: session) act on -- and write audit rows for -- a verdict whose row
        #: can still roll back, and its writes would block on this
        #: transaction's SQLite write lock exactly like the self-deadlock this
        #: design removes.
        self.pending_verdicts: list[VerdictEvent] = []

    async def add_guild_hash(self, guild_id: int, gh: GuildHash) -> GuildHash:
        return await GuildHashRepository(self._session, guild_id).add(gh)

    async def remove_guild_hash(self, guild_id: int, hash_id: str) -> int:
        return await GuildHashRepository(self._session, guild_id).remove(hash_id)

    async def list_guild_hashes(self, guild_id: int) -> list[GuildHash]:
        return list(await GuildHashRepository(self._session, guild_id).list_active())

    async def add_whitelist(self, guild_id: int, entry: GuildWhitelist) -> GuildWhitelist:
        return await WhitelistRepository(self._session, guild_id).add(entry)

    async def get_config(self, guild_id: int) -> dict[str, Any]:
        guild = await GuildRepository(self._session).get(guild_id)
        if guild is None:
            return {}
        return {
            "sensitivity": guild.sensitivity,
            "action_policy": guild.action_policy,
            "mod_queue_threshold": guild.mod_queue_threshold,
            "retention_days": guild.retention_days,
            "locale": guild.locale,
            "safe_mode": guild.safe_mode,
            # Keyed as "review_channel" (not the DB column's "review_channel_id")
            # so this dict's keys always match the field names /config set
            # accepts -- the DB column name is an internal storage detail and
            # must not leak into the user-facing config surface.
            "review_channel": guild.review_channel_id,
            "optin_global_db": guild.optin_global_db,
            "optin_scan_bots": guild.optin_scan_bots,
            "optin_evidence_storage": guild.optin_evidence_storage,
        }

    #: Maps a /config set field name to the Guild ORM attribute it actually
    #: writes, for the one field where they differ. "review_channel" is the
    #: command-facing name (matches validate_config_set and get_config's dict
    #: key); Guild's mapped column is "review_channel_id". Every other field
    #: name is identical to its column name and needs no entry here.
    _FIELD_TO_COLUMN: ClassVar[dict[str, str]] = {"review_channel": "review_channel_id"}

    async def set_config_field(self, guild_id: int, field: str, value: Any) -> None:
        repo = GuildRepository(self._session)
        guild = await repo.get_or_create(guild_id)
        column = self._FIELD_TO_COLUMN.get(field, field)
        # setattr on a mismatched/unmapped name fails silently (it just sets a
        # plain Python attribute with no ORM tracking, discarded on flush) --
        # exactly how "review_channel" broke before _FIELD_TO_COLUMN existed.
        # Guard against that class of bug recurring for any future field.
        if column not in Guild.__table__.columns:
            raise AttributeError(
                f"set_config_field: {field!r} (column {column!r}) is not a mapped "
                "Guild column -- add an entry to _FIELD_TO_COLUMN if the command "
                "field name intentionally differs from the column name."
            )
        setattr(guild, column, value)
        await self._session.flush()

    async def stats_summary(self, guild_id: int) -> dict[str, Any]:
        now = datetime.now(UTC)
        detections = await DetectionRepository(self._session, guild_id).count_in_window(
            now - timedelta(hours=24), now
        )
        return {"detections": detections, "hours": 24}

    async def opt_out_user(self, user_id: int) -> int:
        repo = UserOptoutRepository(self._session)
        await repo.opt_out(user_id)
        return await repo.purge_user(user_id)

    async def purge_guild(self, guild_id: int) -> int:
        return await GuildPurgeRepository(self._session, guild_id).purge()

    async def recent_detection_for(self, guild_id: int, user_id: int) -> int | None:
        recent = await DetectionRepository(self._session, guild_id).list_recent(limit=20)
        for detection in recent:
            if detection.uploader_id == user_id:
                return detection.id
        return None

    async def detection_belongs_to(self, guild_id: int, detection_id: int, user_id: int) -> bool:
        return await DetectionRepository(self._session, guild_id).belongs_to(detection_id, user_id)

    async def open_appeal(self, guild_id: int, detection_id: int, user_id: int) -> int:
        appeal = await AppealRepository(self._session, guild_id).open(
            detection_id=detection_id, user_id=user_id
        )
        return appeal.id

    async def get_appeal(self, guild_id: int, appeal_id: int) -> dict[str, Any] | None:
        from sqlalchemy import select

        from optimus.db.models import Appeal

        stmt = select(Appeal).where(Appeal.guild_id == guild_id, Appeal.id == appeal_id)
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None
        return {"detection_id": row.detection_id, "user_id": row.user_id, "status": row.status}

    async def resolve_appeal(self, guild_id: int, appeal_id: int, *, approved: bool) -> None:
        from sqlalchemy import select

        from optimus.db.models import Appeal

        stmt = select(Appeal).where(Appeal.guild_id == guild_id, Appeal.id == appeal_id)
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        if row is None:
            raise KeyError(appeal_id)
        row.status = "approved" if approved else "denied"
        await self._session.flush()

    async def reverse_detection_action(self, guild_id: int, detection_id: int) -> None:
        repo = DetectionRepository(self._session, guild_id)
        await repo.set_action_taken(detection_id, "reversed")

    async def disable_safe_mode(self, guild_id: int) -> None:
        await GuildRepository(self._session).set_safe_mode(guild_id, False)

    async def local_hash(self, guild_id: int, hash_id: str) -> GuildHash | None:
        return await GuildHashRepository(self._session, guild_id).get(hash_id)

    async def hash_rate_ok(self, user_id: int) -> bool:
        return await self._rl.acquire(f"scamhash:{user_id}", HASH_RATE)

    async def appeal_cooldown_ok(self, user_id: int) -> bool:
        return await self._rl.acquire(
            f"appeal:{user_id}", RateLimit(capacity=1.0, refill_rate=1.0 / self._appeal_cooldown)
        )

    async def audit(
        self, guild_id: int, actor_id: int, action: str, *, target: str | None = None
    ) -> None:
        await ModActionRepository(self._session, guild_id).record(
            actor_id=actor_id, action=action, target=target
        )

    def global_service(self) -> GlobalHashService:
        return GlobalHashService(
            GlobalHashRepository(self._session),
            GlobalSubmitterRepository(self._session),
            self._rl,
            signing_private_key_b64=self._settings.global_signing_private_key,
            signing_public_key_b64=self._settings.global_signing_public_key,
        )

    async def compute_attachment_hashes(self, *, attachment_id: int, url: str) -> AttachmentHashes:
        # No DB access here on purpose -- see the docstring on the protocol
        # method in handlers.InteractionDeps for why this must stay decoupled
        # from any open transaction.
        return await hash_attachment(self._fetch, attachment_id=attachment_id, url=url)

    async def store_attachment_hash(
        self, guild_id: int, *, hashes: AttachmentHashes, added_by: int
    ) -> GuildHash:
        hash_id = f"{hashes.phash:016x}"
        repo = GuildHashRepository(self._session, guild_id)
        existing = await repo.get(hash_id)
        if existing is not None:
            return existing
        return await repo.add(
            GuildHash(
                hash_id=hash_id,
                phash=hashes.phash,
                dhash=hashes.dhash,
                whash=hashes.whash,
                ahash=hashes.ahash,
                mphash=hashes.mphash,
                mdhash=hashes.mdhash,
                mwhash=hashes.mwhash,
                mahash=hashes.mahash,
                source="reviewmsg",
                added_by=added_by,
            )
        )

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
        if self._detection is None:  # pragma: no cover - always wired at app startup
            _log.warning("reviewmsg_no_detection_service", guild_id=guild_id)
            return
        idempotency_key = f"reviewmsg:{guild_id}:{message_id}:{attachment_id}"
        verdict = VerdictEvent(
            correlation_id=get_correlation_id() or idempotency_key,
            occurred_at=datetime.now(UTC),
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            attachment_id=attachment_id,
            uploader_id=uploader_id,
            idempotency_key=idempotency_key,
            verdict=Verdict.SCAM,
            confidence=1.0,
            matched_hash_id=matched_hash_id,
        )
        # Persist through THIS request's session -- the transaction already
        # holds SQLite's write lock (store_attachment_hash flushed an INSERT
        # moments ago), so `submit_confirmed_match`, which opens a second
        # session, would block on our own lock until busy_timeout and raise
        # "database is locked": a self-deadlock every retry reproduces
        # identically. The bus publish is deferred until after commit.
        await self._detection.persist_confirmed_match(self._session, verdict)
        self.pending_verdicts.append(verdict)


class InteractionService:
    """Routes hikari interactions through the pure handlers within a DB scope."""

    def __init__(
        self,
        scope: SessionScope,
        rate_limiter: RateLimiter,
        settings: Settings,
        *,
        fetch: FetchFn | None = None,
        detection: DetectionService | None = None,
    ) -> None:
        self._scope = scope
        self._rl = rate_limiter
        self._settings = settings
        self._fetch = fetch
        self._detection = detection

    async def dispatch_command(self, ctx: InteractionContext) -> InteractionResponse:
        """Run a slash command within a fresh transactional session scope."""
        return await self._run(lambda deps: handle_command(ctx, deps))

    async def dispatch_button(self, ctx: InteractionContext, custom_id: str) -> InteractionResponse:
        """Route a component press to the correct handler (report vs. other)."""
        review = decode_custom_id(custom_id)
        if review is not None:
            return await self._run(lambda deps: handle_review_button(ctx, review, deps))
        component = decode_component_id(custom_id)
        if component is not None:
            return await self._run(
                lambda deps: handle_component(ctx, component.action, component.ref_id, deps)
            )
        return InteractionResponse("button.expired")

    #: Bounded retry budget for interactions that hit a transient SQLite
    #: "database is locked" error. Each interaction gets a *fresh* session on
    #: retry (a failed session's transaction is already rolled back by
    #: :func:`session_scope`'s exception handler on the way out), so retrying
    #: the whole call is safe as long as the handler itself is idempotent --
    #: which every write path here already has to be for message-bus
    #: redelivery (see :meth:`DetectionService._persist`'s idempotency-key
    #: savepoint). This does not attempt to diagnose *why* SQLite reports the
    #: database as locked (WAL requires brief exclusive access to its shared
    #: -shm/-wal files even for readers, and a busy Railway volume can stall
    #: that past the point a single attempt tolerates); it only keeps a rare,
    #: transient lock from surfacing as a failed Discord interaction.
    #:
    #: Production evidence (2026-08-10): three attempts each burned the full
    #: 30s ``sqlite_busy_timeout_ms`` before raising, all with
    #: ``checkedout: 0`` on this process's own pool -- proving the lock
    #: holder is external to this connection pool (not a leaked/stuck
    #: transaction in this process). A single busy_timeout window that long
    #: gives very few *independent* chances for an external, presumably
    #: transient holder to release the lock between attempts: 3 attempts at
    #: 30s each is ~90s of wall time but only 3 windows. Spreading the same
    #: order-of-magnitude wall-clock budget across more, shorter windows
    #: (see ``sqlite_busy_timeout_ms``, lowered accordingly) gives more
    #: opportunities to catch the lock released, and Discord tolerates up to
    #: 15 minutes between deferring and editing the initial response, so
    #: there is ample budget for this without risking "interaction expired".
    _LOCK_RETRY_BACKOFF: ClassVar[BackoffPolicy] = BackoffPolicy(
        base=0.5, multiplier=2.0, max_delay=8.0, max_attempts=6
    )

    async def _run(self, call: Any) -> InteractionResponse:
        attempt = 0
        retry_history: list[dict[str, int | str]] = []

        async def attempt_once() -> InteractionResponse:
            nonlocal attempt
            attempt += 1
            try:
                async with self._scope() as session:
                    deps = DbDeps(
                        session,
                        self._rl,
                        self._settings,
                        fetch=self._fetch,
                        detection=self._detection,
                    )
                    response: InteractionResponse = await call(deps)
            except OperationalError as exc:
                if not _is_sqlite_lock_error(exc):
                    raise _NonRetryableDbError from exc
                diagnostics = _pool_diagnostics(session)
                retry_history.append({"attempt": attempt, **diagnostics})
                _log.warning(
                    "interaction_db_locked_retry",
                    attempt=attempt,
                    max_attempts=self._LOCK_RETRY_BACKOFF.max_attempts,
                    **diagnostics,
                )
                raise
            # The scope has exited: the transaction is committed and its write
            # lock released. Only now is it safe to hand confirmed-scam
            # verdicts to the moderation consumer -- it opens its own DB
            # sessions, whose writes would have contended with (and under
            # SQLite's single-writer lock, deadlocked against) the transaction
            # above. A failed attempt never reaches this line, so a rolled
            # -back attempt's verdicts are discarded with its deps; the
            # retry's fresh attempt re-persists them idempotently.
            if self._detection is not None:
                for verdict in deps.pending_verdicts:
                    await self._detection.publish_confirmed_match(verdict)
            return response

        try:
            return await retry_async(
                attempt_once, self._LOCK_RETRY_BACKOFF, retry_on=(OperationalError,)
            )
        except _NonRetryableDbError as exc:
            assert exc.__cause__ is not None
            raise exc.__cause__ from None
        except OperationalError as exc:
            # All retries exhausted on a genuine (external) lock. Stamp the
            # full attempt history onto the exception so the caller's
            # top-level `_log.exception("interaction_failed")` -- an
            # error-severity log most log viewers surface by default,
            # unlike the warning-level `interaction_db_locked_retry` lines
            # above -- carries enough context to diagnose the incident
            # without having to separately dig up the warning logs.
            exc.add_note(f"lock_retry_history={retry_history!r}")
            raise


class _NonRetryableDbError(Exception):
    """Internal sentinel: an ``OperationalError`` that is not a lock error.

    :func:`InteractionService._run` retries on ``OperationalError`` broadly
    (via :func:`optimus.core.backoff.retry_async`'s type-based filter), but
    only a SQLite "database is locked" message is actually transient. Raising
    this distinct type from inside the retried closure stops the retry loop
    immediately for anything else (e.g. a genuinely broken migration), while
    still letting the original exception surface unchanged to the caller.
    """


def render(response: InteractionResponse, locale: str) -> str:
    """Localize a successful handler response for ephemeral display."""
    return translate(response.i18n_key, locale, **response.params)


def _is_sqlite_lock_error(exc: OperationalError) -> bool:
    """Whether ``exc`` is SQLite's transient ``database is locked`` error.

    ``OperationalError`` covers many unrelated conditions (e.g. a genuinely
    missing table on a broken migration); only the specific SQLite lock
    message is worth a retry, so this checks the wrapped driver message
    rather than treating every ``OperationalError`` as transient.
    """
    return "database is locked" in str(exc.orig).lower()


def _pool_diagnostics(session: Any) -> dict[str, int | str]:
    """Best-effort snapshot of the session's connection pool state.

    A "database is locked" error that survives every retry attempt is not
    supposed to happen under a healthy pool -- either every connection is
    genuinely busy (checkedout == pool size, pointing at a leak or a stuck
    long-lived transaction elsewhere) or something odd is going on with the
    pool configuration itself. This is deliberately defensive: pool types
    that don't track checkout counts (e.g. ``StaticPool`` for ``:memory:``
    databases in tests) shouldn't turn a diagnostic log call into a second
    exception on top of the one being reported.
    """
    try:
        pool = session.get_bind().pool
        return {
            "pool_class": type(pool).__name__,
            "checkedout": pool.checkedout(),
            "checkedin": pool.checkedin(),
        }
    except Exception as exc:
        return {"pool_diagnostics_error": str(exc)}


def _image_attachments(attachments: Any) -> list[tuple[int, str]]:
    """Filter a message's attachments down to ``(id, url)`` pairs for images.

    A scam post usually carries several images (screenshots, QR codes) but can
    also attach unrelated files (e.g. a PDF) that ``attachment_hash`` cannot
    decode -- filtering on ``media_type`` here avoids a doomed fetch attempt
    for those rather than surfacing them as a per-attachment failure below.
    """
    return [
        (int(att.id), att.url) for att in attachments if (att.media_type or "").startswith("image/")
    ]


def _context_menu_context(interaction: Any) -> InteractionContext:
    """Adapt a MESSAGE context-menu interaction ("Review as scam").

    The target message and its author are already resolved by Discord onto
    the interaction itself (``interaction.resolved``), so this needs no REST
    round-trip, unlike the slash-command path in :func:`resolve_review_options`.
    """
    from optimus.services.interactions.commands import REVIEW_MESSAGE_COMMAND

    message = interaction.resolved.messages[interaction.target_id]
    member = interaction.member
    perms = int(member.permissions) if member is not None and member.permissions else 0
    return InteractionContext(
        guild_id=int(interaction.guild_id) if interaction.guild_id is not None else None,
        user_id=int(interaction.user.id),
        member_permissions=perms,
        command=REVIEW_MESSAGE_COMMAND,
        options={
            "channel_id": int(message.channel_id),
            "message_id": int(message.id),
            "author_id": int(message.author.id),
            "attachments": _image_attachments(message.attachments),
        },
        locale=str(getattr(interaction, "locale", "en") or "en"),
    )


async def _resolve_reviewmsg_options(
    ctx: InteractionContext, interaction: Any, *, rest: Any
) -> InteractionContext:
    """Resolve ``/scamhash reviewmsg message:<link-or-id>`` via REST.

    Unlike the context-menu entry point, a slash command only carries the
    moderator-typed string -- the target message must be fetched explicitly.
    A bare id relies on the invoking channel; a full link carries its own
    channel id. Raises :class:`InteractionRejected` (``MESSAGE_NOT_FOUND`` /
    ``FETCH_FAILED``) rather than letting a hikari REST error escape, so the
    normal rejection-to-ephemeral-message path in :func:`run_interaction`
    handles it.
    """
    from optimus.services.interactions.logic import parse_message_reference

    channel_id, message_id = parse_message_reference(str(ctx.options["message"]))
    if channel_id is None:
        channel_id = int(interaction.channel_id)
    try:
        message = await rest.fetch_message(channel_id, message_id)
    except Exception as exc:
        import hikari

        if isinstance(exc, hikari.NotFoundError):
            raise InteractionRejected(CommandError.MESSAGE_NOT_FOUND) from exc
        raise InteractionRejected(CommandError.FETCH_FAILED) from exc
    return InteractionContext(
        guild_id=ctx.guild_id,
        user_id=ctx.user_id,
        member_permissions=ctx.member_permissions,
        command=ctx.command,
        subcommand=ctx.subcommand,
        options={
            "channel_id": int(message.channel_id),
            "message_id": int(message.id),
            "author_id": int(message.author.id),
            "attachments": _image_attachments(message.attachments),
        },
        locale=ctx.locale,
    )


def to_context(interaction: Any) -> InteractionContext:
    """Adapt a hikari command interaction into an :class:`InteractionContext`.

    The member's *effective* permissions come from ``interaction.member`` as
    resolved by Discord (role permissions OR'd, owner short-circuited) — never
    from the command's ``default_member_permissions`` hint. MESSAGE-type
    (context-menu) interactions are delegated to :func:`_context_menu_context`,
    which has an entirely different resolved-data shape from a SLASH command.
    """
    import hikari

    if getattr(interaction, "command_type", hikari.CommandType.SLASH) == hikari.CommandType.MESSAGE:
        return _context_menu_context(interaction)

    interaction_options = interaction.options or []
    options = {option.name: option.value for option in interaction_options}
    subcommand: str | None = None
    # A SUB_COMMAND/SUB_COMMAND_GROUP option carries its selected branch in
    # ``options`` instead of ``value`` — but a parameterless leaf subcommand
    # (e.g. ``/config view``, ``/scamhash list``) also has ``options=None``,
    # identical in shape to a leaf parameter. Discriminate on ``type`` instead
    # of ``options is not None`` so a parameterless subcommand still descends
    # correctly rather than being mistaken for "no more nesting to do".
    while len(interaction_options) == 1 and interaction_options[0].type in (
        hikari.OptionType.SUB_COMMAND,
        hikari.OptionType.SUB_COMMAND_GROUP,
    ):
        selected = interaction_options[0]
        subcommand = selected.name
        interaction_options = selected.options or []
        options = {option.name: option.value for option in interaction_options}
    member = interaction.member
    perms = int(member.permissions) if member is not None and member.permissions else 0
    return InteractionContext(
        guild_id=int(interaction.guild_id) if interaction.guild_id is not None else None,
        user_id=int(interaction.user.id),
        member_permissions=perms,
        command=interaction.command_name,
        subcommand=subcommand,
        options=options,
        locale=str(getattr(interaction, "locale", "en") or "en"),
    )


async def run_interaction(  # pragma: no cover - hikari glue
    service: InteractionService, interaction: Any
) -> str:
    """Handle one interaction end-to-end and return the ephemeral message."""
    import hikari

    with correlation_context():
        try:
            if isinstance(interaction, hikari.CommandInteraction):
                ctx = to_context(interaction)
                locale = ctx.locale
                if ctx.command == "scamhash" and ctx.subcommand == "reviewmsg":
                    ctx = await _resolve_reviewmsg_options(
                        ctx, interaction, rest=interaction.app.rest
                    )
                response = await service.dispatch_command(ctx)
            elif isinstance(interaction, hikari.ComponentInteraction):
                ctx = _component_context(interaction)
                locale = ctx.locale
                response = await service.dispatch_button(ctx, interaction.custom_id)
            else:
                return ""
        except InteractionRejected as rejected:
            return error_message(rejected.reason, locale)
        except Exception:
            _log.exception("interaction_failed")
            return translate("button.expired", locale)
        return render(response, locale)


async def respond_to_interaction(service: InteractionService, interaction: Any) -> None:
    """Defer an interaction before dispatch, then edit in the rendered result."""
    import hikari

    log_context = {
        "interaction_id": str(interaction.id),
        "command_name": getattr(interaction, "command_name", None),
    }
    try:
        await interaction.create_initial_response(
            hikari.ResponseType.DEFERRED_MESSAGE_CREATE,
            flags=hikari.MessageFlag.EPHEMERAL,
        )
    except Exception:
        _log.exception("interaction_defer_failed", **log_context)
        return

    message = await run_interaction(service, interaction)
    if not message:
        return
    try:
        await interaction.edit_initial_response(message)
    except Exception:
        _log.exception("interaction_edit_failed", **log_context)


def _component_context(interaction: Any) -> InteractionContext:  # pragma: no cover - hikari glue
    member = interaction.member
    perms = int(member.permissions) if member is not None and member.permissions else 0
    return InteractionContext(
        guild_id=int(interaction.guild_id) if interaction.guild_id is not None else None,
        user_id=int(interaction.user.id),
        member_permissions=perms,
        command="",
        locale=str(getattr(interaction, "locale", "en") or "en"),
    )


def build_rate_limiter(settings: Settings, redis: object | None) -> RateLimiter:
    """Construct the interactions limiter per ``settings.ratelimit_backend``.

    The Redis backend carries an in-memory fallback that opportunistically
    sweeps idle per-user buckets, so a runtime Redis outage degrades to
    per-process limiting (never crashing the request path) without the
    process-local map growing without bound.
    """
    from optimus.core.ratelimit import build_rate_limiter as _build

    return _build(
        settings,
        redis,
        sweep_interval=settings.interactions_inmemory_sweep_seconds,
    )


def _open_redis(settings: Settings) -> object | None:  # pragma: no cover - boot glue
    try:
        import redis.asyncio as aioredis

        return aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_timeout,
        )
    except Exception:
        _log.warning("redis_unavailable_interactions")
        return None


async def _amain() -> None:  # pragma: no cover - runtime entrypoint
    import hikari

    from optimus.core.config import get_settings
    from optimus.core.health import HealthServer
    from optimus.core.logging import configure_logging
    from optimus.core.readiness import db_check, redis_check
    from optimus.db.engine import create_engine, create_session_factory, session_scope

    settings = get_settings()
    configure_logging(level=settings.log_level, service_name="optimus-interactions")

    engine = create_engine()
    factory = create_session_factory(engine)

    def scope() -> Any:
        return session_scope(factory)

    redis = _open_redis(settings)
    rate_limiter = build_rate_limiter(settings, redis)
    service = InteractionService(scope, rate_limiter, settings)

    health = HealthServer(host=settings.health_host, port=settings.health_port)
    # Interactions serve appeals/commands straight from Postgres, so DB
    # readiness is the dependency that matters most here.
    health.add_readiness_check(db_check(scope), name="postgres")
    if redis is not None:
        health.add_readiness_check(redis_check(redis), name="redis")
    await health.start()

    bot = hikari.GatewayBot(token=settings.discord_token, intents=hikari.Intents.GUILDS)

    @bot.listen(hikari.InteractionCreateEvent)
    async def _on_interaction(event: hikari.InteractionCreateEvent) -> None:
        interaction = event.interaction
        if not isinstance(interaction, hikari.CommandInteraction | hikari.ComponentInteraction):
            return
        await respond_to_interaction(service, interaction)

    try:
        await bot.start()
        await bot.join()
    finally:
        health.set_live(False)
        with contextlib.suppress(Exception):
            await bot.close()
        await health.stop()
        await engine.dispose()


def main() -> None:  # pragma: no cover - console entrypoint
    """Console entrypoint: ``python -m optimus.services.interactions``."""
    import asyncio

    asyncio.run(_amain())
