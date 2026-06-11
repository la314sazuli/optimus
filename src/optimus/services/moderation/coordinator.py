"""Moderation orchestration: verdict -> policy -> boundaries -> action -> audit.

The coordinator ties the pure pieces (:mod:`policy`, :mod:`boundaries`) to the
side-effecting ones (:class:`~optimus.services.moderation.actions.ActionExecutor`,
report posting, audit recording) behind injected callables so the whole flow is
testable without a live gateway or database.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from prometheus_client import Counter

from optimus.contracts.events import Action, VerdictEvent
from optimus.services.moderation.actions import ActionExecutor, ActionRequest, ActionResult
from optimus.services.moderation.boundaries import TargetContext, check_target
from optimus.services.moderation.policy import Decision, PolicyInput, decide
from optimus.services.moderation.review import ReportData

ACTIONS_TAKEN = Counter(
    "optimus_moderation_actions_total",
    "Moderation actions attempted.",
    ["action", "success"],
)
DECISIONS = Counter(
    "optimus_moderation_decisions_total",
    "Policy decisions made.",
    ["decision"],
)
BOUNDARY_REFUSALS = Counter(
    "optimus_moderation_boundary_refusals_total",
    "Punitive actions downgraded by a privilege boundary.",
    ["reason"],
)


@dataclass(frozen=True, slots=True)
class GuildModConfig:
    """The moderation-relevant configuration for one guild."""

    guild_id: int
    configured_action: Action
    mod_queue_threshold: float
    auto_act_threshold: float
    safe_mode: bool
    locale: str = "en"
    guild_name: str = ""
    review_channel_id: int | None = None
    timeout_seconds: int = 3600


#: Resolves a guild's moderation config (Redis-cached / DB-backed at runtime).
ConfigResolver = Callable[[int], Awaitable[GuildModConfig]]
#: Resolves a target's privilege context, or ``None`` if the member is gone.
TargetResolver = Callable[[int, int], Awaitable[TargetContext | None]]
#: Posts a report to the review channel and returns the posted message id.
ReportPoster = Callable[[int, ReportData], Awaitable[int | None]]
#: Persists the action taken + an audit row; returns the detection row id (if any).
AuditRecorder = Callable[[VerdictEvent, str, ActionResult], Awaitable[int | None]]


class ModerationCoordinator:
    """Decides and applies moderation for each verdict."""

    def __init__(
        self,
        *,
        config: ConfigResolver,
        target: TargetResolver,
        executor: ActionExecutor,
        report: ReportPoster,
        audit: AuditRecorder,
    ) -> None:
        self._config = config
        self._target = target
        self._executor = executor
        self._report = report
        self._audit = audit

    async def handle_verdict(self, event: VerdictEvent) -> ActionResult:
        """Process one verdict end-to-end and return the action outcome."""
        cfg = await self._config(event.guild_id)
        outcome = decide(
            PolicyInput(
                verdict=event.verdict,
                confidence=event.confidence,
                configured_action=cfg.configured_action,
                mod_queue_threshold=cfg.mod_queue_threshold,
                auto_act_threshold=cfg.auto_act_threshold,
                safe_mode=cfg.safe_mode,
            )
        )
        DECISIONS.labels(decision=outcome.decision.value).inc()

        action = outcome.action
        decision = outcome.decision

        if decision is Decision.AUTO_ACT and action in (
            Action.DELETE_TIMEOUT,
            Action.DELETE_KICK,
            Action.DELETE_BAN,
        ):
            action, decision = await self._apply_boundaries(event, action, decision)

        if decision is Decision.NONE:
            return ActionResult(Action.NONE, success=True, detail=outcome.reason)

        result = await self._execute(event, cfg, action, decision)
        detection_id = await self._audit(event, action.value, result)
        await self._post_report(event, cfg, action, detection_id)
        return result

    async def _apply_boundaries(
        self, event: VerdictEvent, action: Action, decision: Decision
    ) -> tuple[Action, Decision]:
        ctx = await self._target(event.guild_id, event.uploader_id)
        if ctx is None:
            BOUNDARY_REFUSALS.labels(reason="not_in_guild").inc()
            return Action.REPORT_ONLY, Decision.MOD_QUEUE
        result = check_target(ctx)
        if not result.allowed:
            reason = result.refusal.value if result.refusal else "unknown"
            BOUNDARY_REFUSALS.labels(reason=reason).inc()
            return Action.REPORT_ONLY, Decision.MOD_QUEUE
        return action, decision

    async def _execute(
        self, event: VerdictEvent, cfg: GuildModConfig, action: Action, decision: Decision
    ) -> ActionResult:
        if decision is Decision.MOD_QUEUE or action in (Action.NONE, Action.REPORT_ONLY):
            ACTIONS_TAKEN.labels(action=Action.REPORT_ONLY.value, success="true").inc()
            return ActionResult(Action.REPORT_ONLY, success=True, detail="queued")
        result = await self._executor.execute(
            ActionRequest(
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                message_id=event.message_id,
                uploader_id=event.uploader_id,
                action=action,
                idempotency_key=f"modact:{event.idempotency_key}:{action.value}",
                guild_name=cfg.guild_name,
                locale=cfg.locale,
                timeout_seconds=cfg.timeout_seconds,
            )
        )
        ACTIONS_TAKEN.labels(action=action.value, success=str(result.success).lower()).inc()
        return result

    async def _post_report(
        self,
        event: VerdictEvent,
        cfg: GuildModConfig,
        action: Action,
        detection_id: int | None,
    ) -> None:
        if cfg.review_channel_id is None or detection_id is None:
            return
        await self._report(
            cfg.review_channel_id,
            ReportData(
                detection_id=detection_id,
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                message_id=event.message_id,
                uploader_id=event.uploader_id,
                verdict=event.verdict.value,
                confidence=event.confidence,
                action_taken=action.value,
                matched_hash_id=event.matched_hash_id,
                locale=cfg.locale,
            ),
        )
