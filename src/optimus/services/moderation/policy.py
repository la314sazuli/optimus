"""Pure moderation policy engine.

Given a guild's configuration, a detection verdict and confidence, and the
uploader's context, decide *what should happen*. This module is deliberately
free of Discord, Redis, and database concerns so the full decision matrix is
unit-testable.

Two independent confidence thresholds are honoured:

* ``mod_queue_threshold`` -- at/above this, a detection is surfaced to human
  moderators (``Decision.MOD_QUEUE``).
* ``auto_act_threshold`` -- at/above this, the guild's configured action is
  applied automatically (``Decision.AUTO_ACT``).

``auto_act_threshold`` must be >= ``mod_queue_threshold``; auto-action always
implies the detection also clears the queue bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from optimus.contracts.events import Action, Verdict


class Decision(StrEnum):
    """The disposition of a detection before privilege checks."""

    NONE = "none"
    MOD_QUEUE = "mod_queue"
    AUTO_ACT = "auto_act"


@dataclass(frozen=True, slots=True)
class PolicyInput:
    """Everything the policy engine needs to decide on one detection."""

    verdict: Verdict
    confidence: float
    #: The guild's configured action for an auto-acted scam.
    configured_action: Action
    mod_queue_threshold: float
    auto_act_threshold: float
    #: When true the guild only ever reports; never auto-acts (safe mode).
    safe_mode: bool = False


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    """The decision and the action that should be carried out."""

    decision: Decision
    action: Action
    reason: str


def _decidable(verdict: Verdict) -> bool:
    return verdict in (Verdict.SCAM, Verdict.AMBIGUOUS)


def decide(inp: PolicyInput) -> PolicyOutcome:
    """Map a verdict + confidence + guild config onto a moderation outcome."""
    if inp.auto_act_threshold < inp.mod_queue_threshold:
        raise ValueError("auto_act_threshold must be >= mod_queue_threshold")

    if not _decidable(inp.verdict):
        return PolicyOutcome(Decision.NONE, Action.NONE, f"verdict_{inp.verdict.value}")

    if inp.confidence < inp.mod_queue_threshold:
        return PolicyOutcome(Decision.NONE, Action.NONE, "below_queue_threshold")

    # At/above the queue bar. Decide whether it also clears the auto-act bar.
    auto = inp.confidence >= inp.auto_act_threshold and inp.verdict is Verdict.SCAM
    if not auto:
        return PolicyOutcome(Decision.MOD_QUEUE, Action.REPORT_ONLY, "queued_for_review")

    if inp.safe_mode:
        return PolicyOutcome(Decision.MOD_QUEUE, Action.REPORT_ONLY, "safe_mode_report_only")

    if inp.configured_action in (Action.NONE, Action.REPORT_ONLY):
        return PolicyOutcome(Decision.MOD_QUEUE, Action.REPORT_ONLY, "policy_report_only")

    return PolicyOutcome(Decision.AUTO_ACT, inp.configured_action, "auto_act")
