"""Tests for moderator UX: color-coded embeds, /scamhash recent, help, scanmsg, undo-by-ID."""

from __future__ import annotations

from typing import Any

import pytest

from optimus.db.models import GuildHash
from optimus.services.interactions.attachment_hash import (
    AttachmentHashError,
    AttachmentHashes,
)
from optimus.services.interactions.handlers import (
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    InteractionContext,
    handle_command,
)

MANAGE = 0x20


def _ctx(
    *,
    sub: str,
    options: dict[str, Any] | None = None,
    command: str = "scamhash",
    guild_id: int = 100,
    user_id: int = 200,
) -> InteractionContext:
    return InteractionContext(
        command=command,
        subcommand=sub,
        options=options or {},
        guild_id=guild_id,
        user_id=user_id,
        member_permissions=MANAGE,
    )


class MockDeps:
    """Minimal in-memory InteractionDeps recording side effects."""

    def __init__(self, **flags: Any) -> None:
        self.hashes: dict[str, GuildHash] = {}
        self.confirmed_scams: list[dict[str, Any]] = []
        self.audits: list[tuple[int, int, str, str | None]] = []
        self.reversed: list[int] = []
        self._recent: list[dict[str, Any]] = flags.get("recent", [])
        self._detail: dict[str, Any] | None = flags.get("detail")
        self._last: dict[str, Any] | None = flags.get("last")
        self._attachment_outcomes: dict[int, Any] = flags.get("attachment_outcomes", {})
        self.stored_hashes: list[AttachmentHashes] = []
        self._hash_rate_ok = flags.get("hash_rate_ok", True)

    async def hash_rate_ok(self, user_id: int) -> bool:
        return self._hash_rate_ok

    async def list_guild_hashes(self, guild_id: int) -> list[GuildHash]:
        return list(self.hashes.values())

    async def remove_guild_hash(self, guild_id: int, hash_id: str) -> int:
        return 1 if self.hashes.pop(hash_id, None) is not None else 0

    async def recent_detections(self, guild_id: int, limit: int = 10) -> list[dict[str, Any]]:
        return list(self._recent)

    async def detection_detail(self, guild_id: int, detection_id: int) -> dict[str, Any] | None:
        return self._detail

    async def last_detection(self, guild_id: int) -> dict[str, Any] | None:
        return self._last

    async def reverse_detection_action(self, guild_id: int, detection_id: int) -> None:
        self.reversed.append(detection_id)

    async def audit(
        self, guild_id: int, actor_id: int, action: str, *, target: str | None = None
    ) -> None:
        self.audits.append((guild_id, actor_id, action, target))

    async def compute_attachment_hashes(self, *, attachment_id: int, url: str) -> AttachmentHashes:
        outcome = self._attachment_outcomes.get(attachment_id)
        if isinstance(outcome, Exception):
            raise outcome
        return AttachmentHashes(
            attachment_id=attachment_id,
            url=url,
            phash=attachment_id,
            dhash=attachment_id,
            whash=attachment_id,
            ahash=0,
            mphash=0,
            mdhash=0,
            mwhash=0,
            mahash=0,
            qr_urls=list(outcome.get("qr_urls", [])) if outcome else [],
            ocr_lookalikes=list(outcome.get("lookalikes", [])) if outcome else [],
        )

    async def store_attachment_hash(
        self, guild_id: int, *, hashes: AttachmentHashes, added_by: int
    ) -> GuildHash:
        self.stored_hashes.append(hashes)
        gh = GuildHash(
            hash_id=f"{hashes.phash:016x}",
            phash=hashes.phash,
            dhash=hashes.dhash,
            whash=hashes.whash,
            ahash=0,
            source="scanmsg",
            added_by=added_by,
        )
        self.hashes[gh.hash_id] = gh
        return gh

    async def submit_confirmed_scam(self, guild_id: int, **kwargs: Any) -> None:
        self.confirmed_scams.append({"guild_id": guild_id, **kwargs})

    async def link_campaign(self, guild_id: int, hash_id: str, campaign_id: str) -> None:
        pass

    async def list_campaigns(self, guild_id: int) -> list[tuple[str, int]]:
        return []

    async def list_campaign_hashes(self, guild_id: int) -> list[tuple[str, int]]:
        return []


# --- color mapping -----------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_command_applies_color_from_mapping() -> None:
    """A handler whose response has no explicit color gets one from the mapping."""
    ctx = _ctx(sub="help")
    resp = await handle_command(ctx, MockDeps())
    assert resp.i18n_key == "command.help_text"
    assert resp.color == COLOR_GRAY


@pytest.mark.asyncio
async def test_color_mapping_covers_success_and_warning_paths() -> None:
    # GRAY: empty hash list.
    resp = await handle_command(_ctx(sub="list"), MockDeps())
    assert resp.i18n_key == "command.hash_list_empty"
    assert resp.color == COLOR_GRAY

    # YELLOW: nothing to undo.
    resp = await handle_command(_ctx(sub="undo"), MockDeps(last=None))
    assert resp.i18n_key == "command.undo_nothing"
    assert resp.color == COLOR_YELLOW

    # GREEN: a successful undo.
    detail = {"detection_id": 7, "verdict": "scam", "distance": 3, "action": "delete"}
    resp = await handle_command(_ctx(sub="undo"), MockDeps(last=detail))
    assert resp.i18n_key == "command.undo_done"
    assert resp.color == COLOR_GREEN

    # RED: removing a hash that does not exist.
    resp = await handle_command(_ctx(sub="remove", options={"hash_id": "deadbeef"}), MockDeps())
    assert resp.i18n_key == "command.hash_not_found"
    assert resp.color == COLOR_RED


# --- /scamhash help ----------------------------------------------------------


@pytest.mark.asyncio
async def test_scamhash_help_returns_help_text() -> None:
    resp = await handle_command(_ctx(sub="help"), MockDeps())
    assert resp.i18n_key == "command.help_text"
    assert resp.params == {}


# --- /scamhash recent --------------------------------------------------------


@pytest.mark.asyncio
async def test_scamhash_recent_empty() -> None:
    resp = await handle_command(_ctx(sub="recent"), MockDeps(recent=[]))
    assert resp.i18n_key == "command.recent_empty"
    assert resp.color == COLOR_YELLOW


@pytest.mark.asyncio
async def test_scamhash_recent_with_results() -> None:
    rows = [
        {
            "detection_id": 11,
            "uploader_id": 222,
            "verdict": "scam",
            "action_taken": "delete",
            "created_ts": 1700000000,
        },
        {
            "detection_id": 12,
            "uploader_id": 333,
            "verdict": "clean",
            "action_taken": "none",
            "created_ts": 1700000060,
        },
    ]
    deps = MockDeps(recent=rows)
    resp = await handle_command(_ctx(sub="recent"), deps)
    assert resp.i18n_key == "command.recent_result"
    assert resp.params["count"] == 2
    assert "#11" in resp.params["entries"]
    assert "<@222>" in resp.params["entries"]
    assert "<t:1700000000:R>" in resp.params["entries"]
    assert resp.color == COLOR_GRAY


# --- /scamhash undo with detection_id ---------------------------------------


@pytest.mark.asyncio
async def test_scamhash_undo_by_detection_id_option() -> None:
    detail = {
        "detection_id": 42,
        "verdict": "scam",
        "distance": 5,
        "action": "delete",
    }
    deps = MockDeps(detail=detail)
    ctx = _ctx(sub="undo", options={"detection_id": 42})
    resp = await handle_command(ctx, deps)
    assert resp.i18n_key == "command.undo_done"
    # detection_detail is used (not last_detection) when an id is supplied.
    assert resp.color == COLOR_GREEN
    assert deps.reversed == [42]
    assert deps.audits[0][2] == "scamhash.undo"


@pytest.mark.asyncio
async def test_scamhash_undo_by_id_not_found_falls_back_to_nothing() -> None:
    """A missing specific detection yields undo_nothing, not a crash."""
    deps = MockDeps(detail=None, last=None)
    resp = await handle_command(_ctx(sub="undo", options={"detection_id": 999}), deps)
    assert resp.i18n_key == "command.undo_nothing"
    assert resp.color == COLOR_YELLOW
    assert deps.reversed == []


# --- /scamhash scanmsg -------------------------------------------------------


def _scan_ctx(attachments: list[tuple[int, str]]) -> InteractionContext:
    return _ctx(
        sub="scanmsg",
        options={"author_id": 333, "attachments": attachments, "channel_id": 99, "message_id": 88},
    )


@pytest.mark.asyncio
async def test_scamhash_scanmsg_reports_results_without_storing() -> None:
    outcomes = {
        1: {"qr_urls": ["https://evil.example"], "lookalikes": []},
        2: {
            "qr_urls": [],
            "lookalikes": [{"domain": "0penai.com", "impersonating": "openai.com"}],
        },
    }
    deps = MockDeps(attachment_outcomes=outcomes)
    resp = await handle_command(_scan_ctx([(1, "https://x/1.png"), (2, "https://x/2.png")]), deps)
    assert resp.i18n_key == "command.scanmsg_result"
    summary: str = resp.params["summary"]
    assert "Analyzed 2 image(s) (0 failed)." in summary
    assert "https://evil.example" in summary
    assert "`0penai.com` impersonates `openai.com`" in summary
    assert resp.color == COLOR_YELLOW
    # Preview mode: no hashes stored, no scam submitted, no audit.
    assert deps.stored_hashes == []
    assert deps.confirmed_scams == []
    assert deps.audits == []


@pytest.mark.asyncio
async def test_scamhash_scanmsg_clean_when_no_intel() -> None:
    deps = MockDeps()
    resp = await handle_command(_scan_ctx([(1, "https://x/1.png")]), deps)
    assert resp.i18n_key == "command.scanmsg_result"
    assert "No threats detected." in resp.params["summary"]
    assert deps.stored_hashes == []


@pytest.mark.asyncio
async def test_scamhash_scanmsg_no_images() -> None:
    resp = await handle_command(_scan_ctx([]), MockDeps())
    assert resp.i18n_key == "command.scanmsg_no_images"
    assert resp.color == COLOR_YELLOW


@pytest.mark.asyncio
async def test_scamhash_scanmsg_all_failed() -> None:
    deps = MockDeps(
        attachment_outcomes={
            1: AttachmentHashError("bad"),
            2: AttachmentHashError("bad"),
        }
    )
    resp = await handle_command(_scan_ctx([(1, "https://x/1.png"), (2, "https://x/2.png")]), deps)
    assert resp.i18n_key == "command.scanmsg_all_failed"
    assert resp.params == {"failed": 2}
    assert deps.stored_hashes == []


# --- moderator action buttons -----------------------------------------------


@pytest.mark.asyncio
async def test_scanmsg_attaches_add_scam_and_dismiss_buttons() -> None:
    deps = MockDeps()
    resp = await handle_command(_scan_ctx([(1, "https://x/1.png")]), deps)
    assert len(resp.components) == 2
    assert resp.components[0].label == "Add as scam"
    assert resp.components[1].label == "Dismiss"


@pytest.mark.asyncio
async def test_reviewmsg_with_intel_attaches_undo_and_dismiss_buttons() -> None:
    deps = MockDeps(
        attachment_outcomes={1: {"qr_urls": ["https://evil.example"], "lookalikes": []}}
    )
    ctx = _ctx(
        sub="reviewmsg",
        options={
            "author_id": 333,
            "attachments": [(1, "https://x/1.png")],
            "channel_id": 99,
            "message_id": 88,
        },
    )
    resp = await handle_command(ctx, deps)
    assert resp.i18n_key == "command.reviewmsg_result_with_intel"
    assert len(resp.components) == 2
    assert resp.components[0].label == "Undo"
    assert resp.components[1].label == "Dismiss"


@pytest.mark.asyncio
async def test_mod_button_dismiss_returns_acknowledgment() -> None:
    from optimus.services.interactions.handlers import handle_mod_button
    from optimus.services.interactions.logic import ModAction, ParsedModId

    ctx = _ctx(sub="", command="")
    parsed = ParsedModId(action=ModAction.DISMISS, channel_id=0, message_id=0)
    resp = await handle_mod_button(ctx, parsed, MockDeps())
    assert resp.i18n_key == "button.dismissed"


@pytest.mark.asyncio
async def test_mod_button_undo_reverses_last_detection() -> None:
    from optimus.services.interactions.handlers import handle_mod_button
    from optimus.services.interactions.logic import ModAction, ParsedModId

    deps = MockDeps(last={"detection_id": 42, "verdict": "scam", "action_taken": "delete"})
    ctx = _ctx(sub="", command="")
    parsed = ParsedModId(action=ModAction.UNDO, channel_id=0, message_id=0)
    resp = await handle_mod_button(ctx, parsed, deps)
    assert resp.i18n_key == "command.undo_done"
    assert deps.reversed == [42]
    assert len(deps.audits) == 1
