"""CLI entrypoint for the detection eval harness.

Run with ``python -m benchmarks`` (after ``uv sync``). Prints a readable report
to stdout and optionally writes Markdown and JSON artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.corpus import PERTURBATIONS, build_corpus
from benchmarks.harness import (
    default_thresholds,
    preset_results,
    recommend_operating_point,
    score_corpus,
    sweep_thresholds,
)
from benchmarks.report import build_json, render_markdown, write_json
from optimus.services.detection.matcher import DEFAULT_CANDIDATE_RADIUS


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmarks",
        description="Offline detection-quality evaluation for the scam-image pipeline.",
    )
    parser.add_argument(
        "--campaigns",
        type=int,
        default=None,
        help="cap the number of scam campaigns (default: all built-in campaigns)",
    )
    parser.add_argument(
        "--clean-count",
        type=int,
        default=18,
        help="number of clean negative images (default: 18)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="threshold-sweep granularity (default: 25 steps from 0 to --hi)",
    )
    parser.add_argument(
        "--hi",
        type=float,
        default=0.30,
        help="upper bound of the threshold sweep (default: 0.30)",
    )
    parser.add_argument(
        "--candidate-radius",
        type=int,
        default=DEFAULT_CANDIDATE_RADIUS,
        help=(
            "phash Hamming radius for candidate gathering "
            f"(default: {DEFAULT_CANDIDATE_RADIUS}, the production matcher value)"
        ),
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


def main(argv: list[str] | None = None) -> None:
    """Build the corpus, run the pipeline, and emit the report/artifacts."""
    args = _parse_args(argv)
    corpus = build_corpus(campaigns=args.campaigns, clean_count=args.clean_count)
    scored = score_corpus(corpus, candidate_radius=args.candidate_radius)
    sweep = sweep_thresholds(scored, default_thresholds(steps=args.steps, hi=args.hi))
    presets = preset_results(scored)
    operating_point = recommend_operating_point(sweep)

    report = render_markdown(corpus, scored, sweep, presets, operating_point, PERTURBATIONS)
    print(report)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report + "\n", encoding="utf-8")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        write_json(str(args.json), build_json(corpus, scored, sweep, presets, operating_point))


if __name__ == "__main__":
    main()
