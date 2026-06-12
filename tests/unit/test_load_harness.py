"""Smoke tests for the detection throughput load harness (``benchmarks.load``).

Runs the harness end to end on a tiny corpus through the real detection worker
and asserts the structural invariants it guarantees, so the harness stays
runnable without depending on a full-size load run. Kept fast (a handful of
images at low concurrency); the real baseline sweep lives in
``docs/performance-notes.md``.
"""

from __future__ import annotations

import json

from benchmarks.corpus import build_corpus
from benchmarks.load.__main__ import main
from benchmarks.load.report import build_json, render_report
from benchmarks.load.runner import build_events, run_load


def _tiny_corpus():
    # Two campaigns + a few negatives keeps each smoke run well under a second of
    # CPU beyond the decode subprocesses.
    return build_corpus(campaigns=2, clean_count=4)


def test_build_events_count_and_unique_keys() -> None:
    corpus = _tiny_corpus()
    events = build_events(corpus, 10)
    assert len(events) == 10
    # Unique idempotency keys so none is skipped as a duplicate.
    assert len({e.idempotency_key for e in events}) == 10
    # Payloads are non-empty base64-inlined PNGs.
    assert all(e.data_b64 and e.content_type == "image/png" for e in events)


async def test_run_load_reports_well_formed_metrics() -> None:
    corpus = _tiny_corpus()
    result = await run_load(corpus, concurrency=2, images=6)
    assert result.images == 6
    assert result.concurrency == 2
    assert result.wall_seconds > 0
    assert result.images_per_sec > 0
    # Percentiles are ordered and bounded by the observed max.
    assert 0.0 < result.p50_ms <= result.p95_ms <= result.p99_ms <= result.max_ms
    assert result.peak_rss_mb > 0
    # Every corpus image decodes cleanly -> no non-decisions.
    assert result.non_decisions == 0
    assert result.latencies_ok


async def test_concurrency_overlaps_work() -> None:
    # With a per-image decode subprocess dominating latency, running images
    # concurrently must not take longer in wall time than serial — overlap is the
    # whole point of the harness. Compare total wall time at c=1 vs c=3.
    corpus = _tiny_corpus()
    serial = await run_load(corpus, concurrency=1, images=6)
    parallel = await run_load(corpus, concurrency=3, images=6)
    assert parallel.wall_seconds <= serial.wall_seconds


def test_render_and_json_round_trip() -> None:
    import asyncio

    corpus = _tiny_corpus()
    results = [asyncio.run(run_load(corpus, concurrency=c, images=4)) for c in (1, 2)]
    report = render_report(results, corpus_images=6)
    assert "Detection throughput load test" in report
    assert "Images/s" in report

    payload = build_json(results, corpus_images=6)
    reloaded = json.loads(json.dumps(payload))
    assert len(reloaded["levels"]) == 2
    assert reloaded["levels"][0]["concurrency"] == 1


def test_cli_writes_artifacts(tmp_path, capsys) -> None:
    md_path = tmp_path / "out" / "load.md"
    json_path = tmp_path / "out" / "load.json"
    main(
        [
            "--campaigns",
            "2",
            "--clean-count",
            "4",
            "--concurrency",
            "1",
            "2",
            "--images",
            "4",
            "--markdown",
            str(md_path),
            "--json",
            str(json_path),
        ]
    )
    out = capsys.readouterr().out
    assert "Detection throughput load test" in out
    assert md_path.exists()
    payload = json.loads(json_path.read_text())
    assert len(payload["levels"]) == 2
    assert payload["levels"][0]["images"] == 4
