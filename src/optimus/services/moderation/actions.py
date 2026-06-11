"""Execution of moderation actions against Discord, guarded for safety.

Every punitive action flows through :class:`ActionExecutor`, which layers four
protections over the raw Discord REST calls:

* a **per-guild token bucket** so a noisy guild cannot exhaust the global rate;
* a **circuit breaker** that fails fast when Discord is unhealthy;
* **exponential backoff with jitter** on transient REST failures;
* an **idempotency key** recorded per ``(guild, message, action)`` so a
  redelivered verdict never double-bans.

The Discord surface is abstracted behind :class:`RestActions` so the executor is
testable without a live gateway. DM warnings are rate-limited per user via a
:class:`~optimus.services.moderation.cooldown.Cooldown`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from optimus.contracts.events import Action
from optimus.core.backoff import BackoffPolicy, retry_async
from optimus.core.circuit import CircuitBreaker, CircuitOpenError
from optimus.core.ratelimit import RateLimit, RateLimiter
from optimus.i18n import translate
from optimus.services.moderation.cooldown import Cooldown


def render_dm(locale: str, *, guild: str) -> str:
    """Render the localized DM warning from the i18n catalog (English fallback)."""
    return translate("dm.warning", locale, guild=guild)


class RestActions(Protocol):
    """The minimal Discord REST surface the executor depends on."""

    async def delete_message(self, channel_id: int, message_id: int) -> None: ...

    async def timeout_member(self, guild_id: int, user_id: int, seconds: int) -> None: ...

    async def kick_member(self, guild_id: int, user_id: int, reason: str) -> None: ...

    async def ban_member(self, guild_id: int, user_id: int, reason: str) -> None: ...

    async def unban_member(self, guild_id: int, user_id: int, reason: str) -> None: ...

    async def send_dm(self, user_id: int, content: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """A fully-resolved request to apply one action to one detection."""

    guild_id: int
    channel_id: int
    message_id: int
    uploader_id: int
    action: Action
    idempotency_key: str
    guild_name: str = ""
    locale: str = "en"
    timeout_seconds: int = 3600
    reason: str = "Automated scam-image removal"


@dataclass(frozen=True, slots=True)
class ActionResult:
    """The outcome of attempting an action."""

    action: Action
    success: bool
    detail: str | None = None


class ActionExecutor:
    """Applies moderation actions with rate-limiting, breaker, backoff, idempotency."""

    def __init__(
        self,
        rest: RestActions,
        rate_limiter: RateLimiter,
        *,
        bot_user_id: int,
        rate: RateLimit,
        idempotency_acquire: object,
        dm_cooldown: Cooldown,
        breaker: CircuitBreaker | None = None,
        backoff: BackoffPolicy | None = None,
    ) -> None:
        self._rest = rest
        self._rl = rate_limiter
        self._bot_user_id = bot_user_id
        self._rate = rate
        self._acquire = idempotency_acquire
        self._dm_cooldown = dm_cooldown
        self._breaker = breaker or CircuitBreaker()
        self._backoff = backoff or BackoffPolicy(max_attempts=3)

    async def execute(self, req: ActionRequest) -> ActionResult:
        """Apply ``req`` exactly once, returning the outcome.

        Returns a ``success=False`` result (rather than raising) on rate-limit
        exhaustion, open circuit, idempotency replay, or REST failure so the
        caller can record an audit row in every case.
        """
        if req.action in (Action.NONE, Action.REPORT_ONLY):
            return ActionResult(req.action, success=True, detail="no_enforcement")

        if not await self._acquire(req.idempotency_key):  # type: ignore[operator]
            return ActionResult(req.action, success=False, detail="duplicate")

        if not await self._rl.acquire(f"modact:{req.guild_id}", self._rate):
            return ActionResult(req.action, success=False, detail="rate_limited")

        try:
            await self._breaker.call(lambda: self._run(req))
        except CircuitOpenError:
            return ActionResult(req.action, success=False, detail="circuit_open")
        except Exception as exc:
            return ActionResult(req.action, success=False, detail=f"error:{type(exc).__name__}")
        return ActionResult(req.action, success=True)

    async def _run(self, req: ActionRequest) -> None:
        await retry_async(lambda: self._apply(req), self._backoff)

    async def _apply(self, req: ActionRequest) -> None:
        # The message is always removed first; punitive steps follow.
        await self._rest.delete_message(req.channel_id, req.message_id)
        if req.action is Action.DELETE:
            await self._maybe_dm(req)
            return
        if req.action is Action.DELETE_TIMEOUT:
            await self._rest.timeout_member(req.guild_id, req.uploader_id, req.timeout_seconds)
        elif req.action is Action.DELETE_KICK:
            await self._rest.kick_member(req.guild_id, req.uploader_id, req.reason)
        elif req.action is Action.DELETE_BAN:
            await self._rest.ban_member(req.guild_id, req.uploader_id, req.reason)
        await self._maybe_dm(req)

    async def _maybe_dm(self, req: ActionRequest) -> None:
        if req.uploader_id == self._bot_user_id:
            return
        if not await self._dm_cooldown.acquire(str(req.uploader_id)):
            return
        content = render_dm(req.locale, guild=req.guild_name or str(req.guild_id))
        try:
            await self._rest.send_dm(req.uploader_id, content)
        except Exception:
            return
