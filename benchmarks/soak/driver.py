"""Soak orchestration: boot SimpleApp, drive traffic, sample, attribute leaks.

This is the reusable entrypoint (``python -m benchmarks.soak``). It wires the
*real* :class:`~optimus.app.simple.SimpleApp` with the same stubbed Discord edges
as ``tests/integration/test_simple_mode.py`` (a recording REST that also answers
the target resolver, plus a fake fetcher so ingest never hits the network), then:

* drives mixed clean/scam/transformed images at a configurable rate through the
  real ``message_image.v1`` -> ingest -> detection -> moderation path;
* injects a hostile input every ~30s through the same path;
* fires appeal interactions periodically through the real interactions service;
* samples process health every 30s to a CSV;
* takes tracemalloc snapshots and ``gc`` type-count diffs at start/mid/end so a
  growth signal can be attributed to a source rather than just observed.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import tempfile
import time
import tracemalloc
from collections import Counter as TypeCounter
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.soak import images
from benchmarks.soak.metrics import (
    CsvWriter,
    Sample,
    linear_slope,
    metric_value,
    open_fd_count,
    percentile,
    rss_bytes,
)
from optimus.app.simple import SimpleApp
from optimus.bus.inprocess import InProcessBus
from optimus.contracts.events import (
    SUBJECT_ACTION_RESULT,
    SUBJECT_IMAGE_FETCHED,
    SUBJECT_MESSAGE_IMAGE,
    SUBJECT_VERDICT,
    ActionResultEvent,
    MessageImageEvent,
)
from optimus.core.config import Settings
from optimus.core.idempotency import build_key
from optimus.core.ratelimit import InMemoryRateLimiter, RateLimit
from optimus.db.models import Guild, GuildHash
from optimus.db.repositories import (
    GuildHashRepository,
    GuildRepository,
)
from optimus.hashing.decoder import decode
from optimus.hashing.perceptual import compute_all
from optimus.ingest.fetcher import FetchedImage
from optimus.services.ingest.worker import IngestWorker
from optimus.services.interactions.handlers import InteractionContext
from optimus.services.interactions.logic import Permission
from optimus.services.interactions.service import InteractionService

GUILD_ID = 4242
CAMPAIGN = "camp-soak"
BOT_USER_ID = 999
GUILD_OWNER_ID = 1
_BOT_ROLE_ID = 10
SCAM_UPLOADER_BASE = 9000


def hashes_for(data: bytes) -> dict[str, int]:
    """Decode ``data`` through the real sandboxed decoder and hash its first frame.

    Used to register the scam campaign's perceptual hashes, identical to the
    integration harness's helper but kept local so the benchmark does not import
    from ``tests/`` (which would couple the checked-in driver to test code).
    """
    decoded = decode(data)
    assert decoded is not None, "scam fixture failed to decode"
    return compute_all(decoded.frames[0])


class _SoakRest:
    """A faithful ``RestActions`` double that also answers the target resolver.

    Records every Discord call as a ``(verb, args)`` tuple and, like the
    composition test's REST double, reports the bot as outranking an
    unprivileged uploader so punitive actions are permitted. ``calls`` is capped
    so the recorder itself cannot become the soak's memory leak.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.fail_on: frozenset[str] = frozenset()
        self._call_cap = 5000

    @property
    def verbs(self) -> list[str]:
        return [verb for verb, _ in self.calls]

    def _record(self, verb: str, *args: int) -> None:
        self.calls.append((verb, args))
        if len(self.calls) > self._call_cap:
            del self.calls[: self._call_cap // 2]
        if verb in self.fail_on:
            raise RuntimeError(f"discord_unavailable:{verb}")

    async def delete_message(self, channel_id: int, message_id: int) -> None:
        self._record("delete_message", channel_id, message_id)

    async def timeout_member(self, guild_id: int, user_id: int, seconds: int) -> None:
        self._record("timeout_member", guild_id, user_id, seconds)

    async def kick_member(self, guild_id: int, user_id: int, reason: str) -> None:
        self._record("kick_member", guild_id, user_id)

    async def ban_member(self, guild_id: int, user_id: int, reason: str) -> None:
        self._record("ban_member", guild_id, user_id)

    async def unban_member(self, guild_id: int, user_id: int, reason: str) -> None:
        self._record("unban_member", guild_id, user_id)

    async def send_dm(self, user_id: int, content: str) -> None:
        self._record("send_dm", user_id)

    async def fetch_member(self, guild_id: int, user_id: int) -> _FakeMember:
        return _FakeMember(role_ids=(_BOT_ROLE_ID,) if user_id == BOT_USER_ID else ())

    async def fetch_guild(self, guild_id: int) -> _FakeGuild:
        return _FakeGuild(owner_id=GUILD_OWNER_ID)

    async def fetch_roles(self, guild_id: int) -> tuple[_FakeRole, ...]:
        return (_FakeRole(role_id=_BOT_ROLE_ID, position=5, permissions=0),)


class _FakeMember:
    def __init__(self, role_ids: tuple[int, ...]) -> None:
        self.role_ids = role_ids


class _FakeGuild:
    def __init__(self, owner_id: int) -> None:
        self.owner_id = owner_id


class _FakeRole:
    def __init__(self, role_id: int, position: int, permissions: int) -> None:
        self.id = role_id
        self.position = position
        self.permissions = permissions


class _UrlRegistryFetcher:
    """Maps a per-message URL to its payload so each image flows the real path.

    The driver stashes ``(bytes, content_type)`` under a unique URL before
    publishing the ``message_image`` event; ingest then "fetches" that URL and
    gets exactly those bytes — no network, but the full ingest validation
    (size cap, content handling) and detection decode run unchanged.
    """

    def __init__(self) -> None:
        self._payloads: dict[str, tuple[bytes, str]] = {}

    def register(self, url: str, data: bytes, content_type: str) -> None:
        self._payloads[url] = (data, content_type)

    async def fetch(self, url: str) -> FetchedImage:
        data, content_type = self._payloads.pop(url, (b"", "application/octet-stream"))
        return FetchedImage(data=data, content_type=content_type, final_url=url)


@dataclass
class _Latency:
    """Records end-to-end latency from publish to action_result, by key."""

    sent_at: dict[str, float] = field(default_factory=dict)
    samples_ms: list[float] = field(default_factory=list)
    acked: int = 0

    def mark_sent(self, key: str) -> None:
        self.sent_at[key] = time.perf_counter()

    def mark_done(self, key: str) -> None:
        start = self.sent_at.pop(key, None)
        if start is not None:
            self.samples_ms.append((time.perf_counter() - start) * 1000.0)
            self.acked += 1

    def window(self) -> list[float]:
        # Percentiles over the recent window only, so a transient early spike does
        # not permanently dominate the p95 (we want drift detection, not lifetime).
        return sorted(self.samples_ms[-2000:])


@dataclass
class SoakConfig:
    duration_s: float = 45 * 60
    sample_interval_s: float = 30.0
    target_rate: float = 3.5  # images/sec (mid of 2-5)
    hostile_interval_s: float = 30.0
    appeal_interval_s: float = 45.0
    csv_path: Path = Path("/home/user/workspace/p2_soak_metrics.csv")


def _soak_settings(db_path: Path) -> Settings:
    """Simple-mode settings against a temp SQLite, scheduler intervals compressed.

    The production scheduler intervals (hourly+) would never fire inside a
    45-minute soak, so the maintenance loops are squeezed to tens of seconds —
    this is what exercises "scheduler ticks" as the mission asks while keeping the
    jobs themselves identical.
    """
    return Settings(
        mode="simple",
        simple_database_url=f"sqlite+aiosqlite:///{db_path}",
        discord_token="soak-token",  # noqa: S106
        scheduler_retention_interval=20,
        scheduler_evidence_interval=25,
        scheduler_rollup_interval=30,
        scheduler_index_rebuild_interval=35,
        scheduler_health_interval=15,
        scheduler_retention_purge_interval=40,
    )


async def _seed(app: SimpleApp, scam: bytes) -> None:
    async with app._scope() as session:
        await GuildRepository(session).upsert(
            Guild(
                guild_id=GUILD_ID,
                action_policy="delete_ban",
                mod_queue_threshold=0.5,
                sensitivity="balanced",
                review_channel_id=None,
            )
        )
        h = hashes_for(scam)
        await GuildHashRepository(session, GUILD_ID).add(
            GuildHash(
                guild_id=GUILD_ID,
                hash_id=CAMPAIGN,
                phash=h["phash"],
                dhash=h["dhash"],
                whash=h["whash"],
                ahash=h["ahash"],
                source="local",
                status="active",
            )
        )


def _soak_ingest_worker(app: SimpleApp, fetcher: _UrlRegistryFetcher) -> IngestWorker:
    return IngestWorker(
        fetcher.fetch,
        InMemoryRateLimiter(),
        rate=RateLimit(capacity=10_000.0, refill_rate=10_000.0),
        max_inline_bytes=app.settings.ingest_max_inline_bytes,
    )


@dataclass
class _GcSnapshot:
    label: str
    elapsed_s: float
    rss_mb: float
    type_counts: dict[str, int]
    tracemalloc_top: list[str]


@dataclass
class SoakSummary:
    """Post-run aggregates the pass-criteria judgement and report are built from."""

    samples: int
    rss_slope_mb_per_s: float
    fd_slope_per_s: float
    task_slope_per_s: float
    memstore_slope_per_s: float
    p95_slope_ms_per_s: float
    rss_first: float
    rss_last: float
    fds_first: float
    fds_last: float
    tasks_first: float
    tasks_last: float
    memstore_first: float
    memstore_last: float
    p95_last: float
    images_sent: int
    images_acked: int
    hostile_sent: int
    hostile_crashed_publisher: int
    appeals_run: int
    appeals_ok: int
    errors: dict[str, int]
    snapshots: list[_GcSnapshot]
    rows: list[dict[str, str]]


def _gc_type_counts(top: int = 30) -> dict[str, int]:
    counts: TypeCounter[str] = TypeCounter()
    for obj in gc.get_objects():
        counts[type(obj).__name__] += 1
    return dict(counts.most_common(top))


def _tracemalloc_top(snapshot: tracemalloc.Snapshot, limit: int = 15) -> list[str]:
    stats = snapshot.statistics("lineno")[:limit]
    return [f"{s.size / 1024:.1f} KiB  {s.count}  {s.traceback}" for s in stats]


class SoakDriver:
    """Owns one soak run end to end."""

    def __init__(self, cfg: SoakConfig) -> None:
        self.cfg = cfg
        self.fetcher = _UrlRegistryFetcher()
        self.latency = _Latency()
        self.errors: TypeCounter[str] = TypeCounter()
        self.images_sent = 0
        self.hostile_sent = 0
        self.hostile_bad = 0
        self.appeals_run = 0
        self.appeals_ok = 0
        self._seq = 0
        self.snapshots: list[_GcSnapshot] = []
        self._start = 0.0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def _publish_image(self, app: SimpleApp, data: bytes, content_type: str) -> None:
        seq = self._next_seq()
        url = f"https://cdn.soak/{seq}.png"
        self.fetcher.register(url, data, content_type)
        # The pipeline's idempotency key is build_key(message_id, attachment_id);
        # a unique message_id per send keeps every image a distinct unit of work
        # (so the idempotency keyspace grows exactly as production would). Key the
        # latency tracker on the same value the action_result carries back.
        self.latency.mark_sent(build_key(seq, 1))
        event = MessageImageEvent(
            correlation_id=f"soak-{seq}",
            occurred_at=datetime.now(UTC),
            guild_id=GUILD_ID,
            channel_id=222,
            message_id=seq,
            attachment_id=1,
            uploader_id=SCAM_UPLOADER_BASE + (seq % 50),
            url=url,
            filename=f"{seq}.png",
            content_type=content_type,
        )
        await app.bus.publish(SUBJECT_MESSAGE_IMAGE, event)

    async def _traffic_lane(self, app: SimpleApp, stop: asyncio.Event) -> None:
        """Sustain ~target_rate img/s of mixed clean/scam/transformed images."""
        interval = 1.0 / self.cfg.target_rate
        i = 0
        while not stop.is_set():
            i += 1
            kind = i % 5
            try:
                if kind in (0, 1):
                    data = images.clean_png(i)
                elif kind in (2, 3):
                    data = images.transformed_png(i)
                else:
                    data = images.scam_png()
                await self._publish_image(app, data, "image/png")
                self.images_sent += 1
            except Exception as exc:  # pragma: no cover - defensive
                self.errors[f"traffic:{type(exc).__name__}"] += 1
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval)

    async def _hostile_lane(self, app: SimpleApp, stop: asyncio.Event) -> None:
        """Inject one hostile payload every ~hostile_interval_s through the path."""
        i = 0
        while not stop.is_set():
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=self.cfg.hostile_interval_s)
            if stop.is_set():
                break
            builder = images.HOSTILE_BUILDERS[i % len(images.HOSTILE_BUILDERS)]
            i += 1
            try:
                data, content_type = builder()
                await self._publish_image(app, data, content_type)
                self.hostile_sent += 1
            except Exception as exc:
                # A hostile input that crashes the *publisher* is itself a failure
                # of the "process unaffected" criterion — record it loudly.
                self.hostile_bad += 1
                self.errors[f"hostile:{type(exc).__name__}"] += 1

    async def _appeal_lane(self, interactions: InteractionService, stop: asyncio.Event) -> None:
        """Fire an /appeal interaction periodically through the real service."""
        while not stop.is_set():
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=self.cfg.appeal_interval_s)
            if stop.is_set():
                break
            self.appeals_run += 1
            # Appeal as a recently-banned uploader; the cooldown limiter means most
            # repeats are no-ops, which is itself a realistic interaction load.
            uploader = SCAM_UPLOADER_BASE + (self.appeals_run % 50)
            ctx = InteractionContext(
                guild_id=GUILD_ID,
                user_id=uploader,
                member_permissions=int(Permission.MANAGE_GUILD),
                command="appeal",
            )
            try:
                await interactions.dispatch_command(ctx)
                self.appeals_ok += 1
            except Exception as exc:
                self.errors[f"appeal:{type(exc).__name__}"] += 1

    def _take_snapshot(self, label: str) -> None:
        gc.collect()
        snap = tracemalloc.take_snapshot()
        self.snapshots.append(
            _GcSnapshot(
                label=label,
                elapsed_s=time.perf_counter() - self._start,
                rss_mb=rss_bytes() / (1024 * 1024),
                type_counts=_gc_type_counts(),
                tracemalloc_top=_tracemalloc_top(snap),
            )
        )

    def _sample(self, app: SimpleApp, db_path: Path) -> Sample:
        bus: InProcessBus = app.bus
        elapsed = time.perf_counter() - self._start

        def qdepth(subject: str) -> int:
            consumers = bus._consumers.get(subject, [])
            return sum(c.queue.qsize() for c in consumers)

        wal = db_path.with_suffix(db_path.suffix + "-wal")
        win = self.latency.window()
        store_keys = len(app.store._data)
        return Sample(
            elapsed_s=elapsed,
            rss_mb=rss_bytes() / (1024 * 1024),
            open_fds=open_fd_count(),
            asyncio_tasks=len(asyncio.all_tasks()),
            sqlite_bytes=db_path.stat().st_size if db_path.exists() else 0,
            sqlite_wal_bytes=wal.stat().st_size if wal.exists() else 0,
            memstore_keys=store_keys,
            q_message_image=qdepth(SUBJECT_MESSAGE_IMAGE),
            q_image_fetched=qdepth(SUBJECT_IMAGE_FETCHED),
            q_verdict=qdepth(SUBJECT_VERDICT),
            images_sent=self.images_sent,
            images_acked=self.latency.acked,
            p50_ms=percentile(win, 50),
            p95_ms=percentile(win, 95),
            p99_ms=percentile(win, 99),
            errors_total=sum(self.errors.values()),
            ingest_rejected=metric_value("optimus_ingest_images_rejected_total"),
            detection_payload_rejected=metric_value("optimus_detection_payload_rejected_total"),
            verdicts_clean=metric_value("optimus_detection_verdicts_total", {"verdict": "clean"}),
            verdicts_scam=metric_value("optimus_detection_verdicts_total", {"verdict": "scam"}),
            verdicts_non_decision=metric_value(
                "optimus_detection_verdicts_total", {"verdict": "non_decision"}
            ),
            bus_dropped=metric_value("optimus_bus_messages_dropped_total"),
            bus_naked=metric_value("optimus_bus_messages_naked_total"),
        )

    async def run(self) -> SoakSummary:
        tracemalloc.start(25)
        self._start = time.perf_counter()
        db_path = Path(tempfile.gettempdir()) / "optimus_soak.db"
        for p in (db_path, db_path.with_suffix(".db-wal"), db_path.with_suffix(".db-shm")):
            with suppress(OSError):
                p.unlink()

        settings = _soak_settings(db_path)
        rest = _SoakRest()
        app = await SimpleApp.build(settings, rest=rest, bot_user_id=BOT_USER_ID)
        await app.dispatcher.start()
        app.ingest_worker = _soak_ingest_worker(app, self.fetcher)
        await _seed(app, images.scam_png())

        # Capture end-to-end completion: register a consumer on action_result so we
        # can time publish -> enforcement. Simple mode emits this subject but wires
        # no consumer, so this is the soak's own terminal hook.
        async def _on_result(event: ActionResultEvent) -> None:
            self.latency.mark_done(event.idempotency_key)

        result_task = app.bus.run(
            SUBJECT_ACTION_RESULT,
            durable="soak-result",
            model=ActionResultEvent,
            handler=_on_result,
            stop_event=app._stop,
        )

        interactions = InteractionService(
            app._scope,
            InMemoryRateLimiter(),
            settings,
        )

        app.start_pipeline()

        stop = asyncio.Event()
        lanes = [
            asyncio.create_task(self._traffic_lane(app, stop)),
            asyncio.create_task(self._hostile_lane(app, stop)),
            asyncio.create_task(self._appeal_lane(interactions, stop)),
        ]

        # Warm up briefly, then take the first attribution snapshot.
        await asyncio.sleep(min(20.0, self.cfg.duration_s / 10))
        self._take_snapshot("start")
        mid_at = self.cfg.duration_s / 2

        summary: SoakSummary
        try:
            with CsvWriter(self.cfg.csv_path) as csv:
                next_sample = self.cfg.sample_interval_s
                mid_taken = False
                while True:
                    elapsed = time.perf_counter() - self._start
                    if elapsed >= self.cfg.duration_s:
                        break
                    if not mid_taken and elapsed >= mid_at:
                        self._take_snapshot("mid")
                        mid_taken = True
                    if elapsed >= next_sample:
                        csv.write(self._sample(app, db_path))
                        next_sample += self.cfg.sample_interval_s
                    await asyncio.sleep(0.5)
                # Final sample + snapshot before teardown.
                self._take_snapshot("end")
                final = self._sample(app, db_path)
                csv.write(final)
        finally:
            stop.set()
            await asyncio.gather(*lanes, return_exceptions=True)
            # Let in-flight work drain so the post-run summary is not skewed by
            # queued-but-unprocessed images.
            await asyncio.sleep(2.0)
            result_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await result_task
            await app.aclose()

        summary = self._build_summary()
        tracemalloc.stop()
        return summary

    def _build_summary(self) -> SoakSummary:
        import csv as _csv

        rows: list[dict[str, str]] = []
        with self.cfg.csv_path.open(encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))

        def col(name: str) -> list[float]:
            return [float(r[name]) for r in rows]

        # Slopes are computed over the post-warmup tail (skip the first 2 samples)
        # so allocator/JIT warmup does not masquerade as a leak.
        tail = rows[2:] if len(rows) > 4 else rows
        t = [float(r["elapsed_s"]) for r in tail]
        return SoakSummary(
            samples=len(rows),
            rss_slope_mb_per_s=linear_slope(t, [float(r["rss_mb"]) for r in tail]),
            fd_slope_per_s=linear_slope(t, [float(r["open_fds"]) for r in tail]),
            task_slope_per_s=linear_slope(t, [float(r["asyncio_tasks"]) for r in tail]),
            memstore_slope_per_s=linear_slope(t, [float(r["memstore_keys"]) for r in tail]),
            p95_slope_ms_per_s=linear_slope(t, [float(r["p95_ms"]) for r in tail]),
            rss_first=col("rss_mb")[0] if rows else 0.0,
            rss_last=col("rss_mb")[-1] if rows else 0.0,
            fds_first=col("open_fds")[0] if rows else 0.0,
            fds_last=col("open_fds")[-1] if rows else 0.0,
            tasks_first=col("asyncio_tasks")[0] if rows else 0.0,
            tasks_last=col("asyncio_tasks")[-1] if rows else 0.0,
            memstore_first=col("memstore_keys")[0] if rows else 0.0,
            memstore_last=col("memstore_keys")[-1] if rows else 0.0,
            p95_last=col("p95_ms")[-1] if rows else 0.0,
            images_sent=self.images_sent,
            images_acked=self.latency.acked,
            hostile_sent=self.hostile_sent,
            hostile_crashed_publisher=self.hostile_bad,
            appeals_run=self.appeals_run,
            appeals_ok=self.appeals_ok,
            errors=dict(self.errors),
            snapshots=self.snapshots,
            rows=rows,
        )


def _print_summary(s: SoakSummary) -> None:
    print("\n===== SOAK SUMMARY =====")
    print(f"samples:            {s.samples}")
    print(f"images sent/acked:  {s.images_sent} / {s.images_acked}")
    print(
        f"hostile sent:       {s.hostile_sent} (publisher crashes: {s.hostile_crashed_publisher})"
    )
    print(f"appeals run/ok:     {s.appeals_run} / {s.appeals_ok}")
    print(f"errors:             {s.errors}")
    print(
        f"RSS  {s.rss_first:.1f} -> {s.rss_last:.1f} MB "
        f"(slope {s.rss_slope_mb_per_s * 60:.4f} MB/min)"
    )
    print(f"fds  {s.fds_first:.0f} -> {s.fds_last:.0f} (slope {s.fd_slope_per_s * 60:.4f}/min)")
    print(
        f"tasks {s.tasks_first:.0f} -> {s.tasks_last:.0f} (slope {s.task_slope_per_s * 60:.4f}/min)"
    )
    print(
        f"memstore keys {s.memstore_first:.0f} -> {s.memstore_last:.0f} "
        f"(slope {s.memstore_slope_per_s * 60:.2f}/min)"
    )
    print(f"p95 last {s.p95_last:.2f} ms (slope {s.p95_slope_ms_per_s * 60:.5f} ms/min)")
    print("\n--- attribution snapshots ---")
    for snap in s.snapshots:
        print(f"\n[{snap.label}] t={snap.elapsed_s:.0f}s  rss={snap.rss_mb:.1f}MB")
        print("  top gc types:", dict(list(snap.type_counts.items())[:8]))
    if len(s.snapshots) >= 2:
        first, last = s.snapshots[0], s.snapshots[-1]
        print("\n--- gc type growth (start -> end) ---")
        diffs = {
            name: last.type_counts.get(name, 0) - first.type_counts.get(name, 0)
            for name in set(first.type_counts) | set(last.type_counts)
        }
        for name, delta in sorted(diffs.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {name:>24}: {delta:+d}")
        print("\n--- tracemalloc top (end) ---")
        for line in last.tracemalloc_top[:10]:
            print(f"  {line}")


def _parse_args(argv: list[str] | None) -> SoakConfig:
    p = argparse.ArgumentParser(description="Optimus simple-mode soak harness")
    p.add_argument("--duration-s", type=float, default=45 * 60)
    p.add_argument("--sample-interval-s", type=float, default=30.0)
    p.add_argument("--rate", type=float, default=3.5, help="images/sec (2-5 realistic)")
    p.add_argument("--hostile-interval-s", type=float, default=30.0)
    p.add_argument("--appeal-interval-s", type=float, default=45.0)
    p.add_argument("--csv", type=Path, default=Path("/home/user/workspace/p2_soak_metrics.csv"))
    a = p.parse_args(argv)
    return SoakConfig(
        duration_s=a.duration_s,
        sample_interval_s=a.sample_interval_s,
        target_rate=a.rate,
        hostile_interval_s=a.hostile_interval_s,
        appeal_interval_s=a.appeal_interval_s,
        csv_path=a.csv,
    )


async def amain(argv: list[str] | None = None) -> SoakSummary:
    cfg = _parse_args(argv)
    driver = SoakDriver(cfg)
    summary = await driver.run()
    _print_summary(summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    asyncio.run(amain(argv))


if __name__ == "__main__":
    main()
