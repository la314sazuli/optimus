"""Synthetic load driver: publish ``image_fetched.v1`` events onto the bus.

Used for chaos/soak testing the distributed stack: it injects a steady stream of
valid, decodable images straight into NATS JetStream via the repo's own
:class:`~optimus.bus.nats.EventBus`, so the detection service exercises its full
decode -> hash -> match -> persist -> publish path while dependencies are killed
and restored around it.

The images are tiny, deterministic, decodable PNGs (so detection resolves them
to CLEAN verdicts rather than NON_DECISION decode failures). Each event carries a
unique idempotency key so nothing is deduped away.

Run inside a one-off container on the compose network (NATS is not published to
the host)::

    docker compose run --rm --no-deps \
        -e OPTIMUS_NATS_URL=nats://nats:4222 \
        --entrypoint python ingest scripts/chaos_load.py --rate 5 --duration 600

Prints a one-line JSON summary on exit (sent / failed counts) so the harness can
reconcile it against what detection persisted.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
import uuid
from datetime import UTC, datetime

from PIL import Image

from optimus.bus.nats import EventBus
from optimus.contracts.events import SUBJECT_IMAGE_FETCHED, ImageFetchedEvent
from optimus.core.config import get_settings


def _make_png(seed: int) -> bytes:
    """A tiny, valid, decodable PNG whose pixel content varies with ``seed``.

    Varying the content keeps each image distinct enough to be realistic; the
    decoder only needs a real raster, not a particular one.
    """
    img = Image.new("RGB", (16, 16), color=(seed % 256, (seed * 7) % 256, (seed * 13) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _event(seq: int, guild_id: int, run_id: str) -> ImageFetchedEvent:
    raw = _make_png(seq)
    import hashlib

    sha = hashlib.sha256(raw).hexdigest()
    # The key is unique per run: reusing keys across runs would let JetStream's
    # publish-dedup window collapse this run's events into earlier ones, which
    # would silently understate the delivered count during chaos accounting.
    key = f"chaos-{run_id}-{seq}"
    return ImageFetchedEvent(
        correlation_id=key,
        occurred_at=datetime.now(UTC),
        guild_id=guild_id,
        channel_id=1,
        message_id=seq,
        attachment_id=seq,
        uploader_id=42,
        idempotency_key=key,
        content_type="image/png",
        size_bytes=len(raw),
        sha256=sha,
        data_b64=base64.b64encode(raw).decode("ascii"),
    )


async def _run(rate: float, duration: float, guild_id: int, run_id: str) -> dict[str, object]:
    settings = get_settings()
    bus, nc = await EventBus.connect(settings.nats_url)
    await bus.ensure_stream(duplicate_window=settings.bus_duplicate_window_seconds)

    sent = 0
    failed = 0
    interval = 1.0 / rate if rate > 0 else 0.0
    deadline = time.monotonic() + duration
    seq = 0
    try:
        while time.monotonic() < deadline:
            tick = time.monotonic()
            seq += 1
            key = f"chaos-{run_id}-{seq}"
            try:
                await bus.publish(SUBJECT_IMAGE_FETCHED, _event(seq, guild_id, run_id), msg_id=key)
                sent += 1
            except Exception as exc:  # chaos driver tolerates bus outages
                failed += 1
                print(f"publish failed seq={seq}: {exc!r}", file=sys.stderr)
            sleep_for = interval - (time.monotonic() - tick)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
    finally:
        with __import__("contextlib").suppress(Exception):
            await nc.drain()
    return {
        "sent": sent,
        "failed": failed,
        "highest_seq": seq,
        "run_id": run_id,
        "guild_id": guild_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=5.0, help="images per second")
    parser.add_argument("--duration", type=float, default=600.0, help="seconds to run")
    parser.add_argument("--guild-id", type=int, default=1, help="guild id to stamp on events")
    parser.add_argument("--run-id", default=None, help="unique tag for this run (default: random)")
    args = parser.parse_args()
    run_id = args.run_id or uuid.uuid4().hex[:8]
    result = asyncio.run(_run(args.rate, args.duration, args.guild_id, run_id))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
