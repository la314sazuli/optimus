"""CLI entrypoint for the detection throughput load test.

Run with ``python -m benchmarks.load`` (after ``uv sync``). Sweeps one or more
concurrency levels, printing a summary table to stdout and optionally writing
Markdown and JSON artifacts.

Examples::

    python -m benchmarks.load --concurrency 1 4 8 --images 200
    python -m benchmarks.load --campaigns 6 --clean-count 18 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from benchmarks.corpus import build_corpus
from benchmarks.load.report import build_json, render_report, write_json
from benchmarks.load.runner import LoadResult, run_load


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmarks.load",
        description="Throughput load test for the detection worker over a synthetic corpus.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="one or more in-flight concurrency levels to sweep (default: 1 4 8)",
    )
    parser.add_argument(
        "--images",
        type=int,
        default=200,
        help="number of images to push per concurrency level (default: 200)",
    )
    parser.add_argument(
        "--campaigns",
        type=int,
        default=None,
        help="cap the number of scam campaigns seeding the index (default: all built-in)",
    )
    parser.add_argument(
        "--clean-count",
        type=int,
        default=18,
        help="number of clean negative images in the corpus pool (default: 18)",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="write the report to this Markdown file as well as stdout",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write a JSON artifact of the results to this path",
    )
    return parser.parse_args(argv)


async def _run_levels(
    *, concurrency: list[int], images: int, campaigns: int | None, clean_count: int
) -> tuple[list[LoadResult], int]:
    """Build the corpus once and run each concurrency level over it."""
    corpus = build_corpus(campaigns=campaigns, clean_count=clean_count)
    corpus_images = len(list(corpus.all_images()))
    results: list[LoadResult] = []
    for level in concurrency:
        results.append(await run_load(corpus, concurrency=level, images=images))
    return results, corpus_images


def main(argv: list[str] | None = None) -> None:
    """Parse args, run the load sweep, and emit the report/artifacts."""
    args = _parse_args(argv)
    results, corpus_images = asyncio.run(
        _run_levels(
            concurrency=args.concurrency,
            images=args.images,
            campaigns=args.campaigns,
            clean_count=args.clean_count,
        )
    )

    report = render_report(results, corpus_images=corpus_images)
    print(report)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report + "\n", encoding="utf-8")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        write_json(str(args.json), build_json(results, corpus_images=corpus_images))


if __name__ == "__main__":
    main()
