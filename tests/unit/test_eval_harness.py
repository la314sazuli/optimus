"""Smoke test for the detection eval harness (``benchmarks`` package).

Runs a tiny synthetic corpus end to end through the real pipeline and asserts
the structural invariants the harness guarantees, so the eval stays runnable in
CI without depending on the full benchmark size.
"""

from __future__ import annotations

import json

from benchmarks.__main__ import main
from benchmarks.corpus import PERTURBATIONS, build_corpus
from benchmarks.harness import (
    NO_MATCH_SCORE,
    default_thresholds,
    evaluate_threshold,
    perturbation_recall,
    preset_results,
    recommend_operating_point,
    score_corpus,
    sweep_thresholds,
)
from benchmarks.report import build_json, render_markdown


def _tiny_corpus():
    # Two campaigns + a handful of negatives keeps the smoke test sub-second.
    return build_corpus(campaigns=2, perturbations=PERTURBATIONS, clean_count=4)


def test_corpus_is_deterministic() -> None:
    a = build_corpus(campaigns=2, clean_count=4)
    b = build_corpus(campaigns=2, clean_count=4)
    assert [i.name for i in a.all_images()] == [i.name for i in b.all_images()]
    # Same pixels -> same hashes -> same scores.
    assert [s.best_score for s in score_corpus(a)] == [s.best_score for s in score_corpus(b)]


def test_bases_match_themselves_exactly() -> None:
    corpus = _tiny_corpus()
    scored = {s.image.name: s for s in score_corpus(corpus)}
    for base in corpus.bases:
        # A base image indexed against itself has zero distance -> score 0.
        assert scored[base.name].best_score == 0.0
        assert scored[base.name].matched_campaign == base.campaign


def test_clean_images_are_not_flagged_at_default_thresholds() -> None:
    corpus = _tiny_corpus()
    scored = score_corpus(corpus)
    for r in sweep_thresholds(scored, default_thresholds()):
        assert r.fp == 0, f"clean image flagged at threshold {r.threshold}"


def test_recall_is_monotonic_in_threshold() -> None:
    corpus = _tiny_corpus()
    scored = score_corpus(corpus)
    results = sweep_thresholds(scored, default_thresholds())
    recalls = [r.recall for r in results]
    assert recalls == sorted(recalls), "recall must not decrease as threshold rises"


def test_resize_and_recompress_are_caught() -> None:
    corpus = _tiny_corpus()
    scored = score_corpus(corpus)
    # The recommended (zero-FP) operating point should catch robust perturbations.
    op = recommend_operating_point(sweep_thresholds(scored, default_thresholds()))
    recall = perturbation_recall(scored, op.result.threshold)
    assert recall["resize"][0] == recall["resize"][1]
    assert recall["recompress"][0] == recall["recompress"][1]
    assert recall["base"][0] == recall["base"][1]


def test_flip_is_not_caught() -> None:
    # Perceptual hashes are not horizontal-flip invariant; this documents the
    # known limitation as an executable expectation.
    corpus = _tiny_corpus()
    scored = score_corpus(corpus)
    recall = perturbation_recall(scored, default_thresholds()[-1])
    assert recall["flip"][0] == 0


def test_no_match_image_gets_sentinel_score() -> None:
    corpus = _tiny_corpus()
    scored = {s.image.name: s for s in score_corpus(corpus)}
    # Clean negatives are far from every base -> no candidate -> sentinel.
    assert any(s.best_score == NO_MATCH_SCORE for s in scored.values())


def test_threshold_metrics_are_well_formed() -> None:
    corpus = _tiny_corpus()
    scored = score_corpus(corpus)
    r = evaluate_threshold(scored, 0.12)
    total = r.tp + r.fp + r.tn + r.fn
    assert total == len(list(corpus.all_images()))
    assert 0.0 <= r.precision <= 1.0
    assert 0.0 <= r.recall <= 1.0
    assert 0.0 <= r.f1 <= 1.0


def test_preset_results_cover_all_presets() -> None:
    corpus = _tiny_corpus()
    presets = preset_results(score_corpus(corpus))
    assert {p.name for p in presets} == {"strict", "balanced", "permissive"}
    for p in presets:
        assert p.flagged.fp == 0


def test_report_and_json_render(tmp_path) -> None:
    corpus = _tiny_corpus()
    scored = score_corpus(corpus)
    sweep = sweep_thresholds(scored, default_thresholds())
    presets = preset_results(scored)
    op = recommend_operating_point(sweep)

    md = render_markdown(corpus, scored, sweep, presets, op, PERTURBATIONS)
    assert "Detection evaluation" in md
    assert "Recommended operating point" in md

    payload = build_json(corpus, scored, sweep, presets, op)
    # Round-trips and contains the headline sections.
    reloaded = json.loads(json.dumps(payload))
    assert reloaded["operating_point"]["fpr"] == 0.0
    assert reloaded["perturbation_recall"]["flip"]["caught"] == 0


def test_cli_writes_artifacts(tmp_path, capsys) -> None:
    md_path = tmp_path / "out" / "report.md"
    json_path = tmp_path / "out" / "report.json"
    main(
        [
            "--campaigns",
            "2",
            "--clean-count",
            "4",
            "--steps",
            "6",
            "--markdown",
            str(md_path),
            "--json",
            str(json_path),
        ]
    )
    out = capsys.readouterr().out
    assert "Detection evaluation" in out
    assert md_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["corpus"]["campaigns"] == 2
