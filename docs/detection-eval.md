# Detection-quality evaluation harness

The `benchmarks` package measures the offline detection quality of the
perceptual-hash + BK-tree + ensemble pipeline: how well it re-identifies a known
scam image after realistic re-share edits (recall), how often it flags benign
uploads (false-positive rate), and how those trade off as the match threshold
moves. It runs the **real** detection code — the same hash functions, BK-tree
candidate gathering, and ensemble scoring the bot uses in production — over a
deterministic synthetic image corpus, so the numbers reflect the shipped logic
rather than a re-implementation.

This complements [`docs/eval/baseline.md`](eval/baseline.md), which scores the
small on-disk fixture set at the three shipped presets. The harness here is
richer (more perturbation kinds, a full threshold sweep, per-perturbation
breakdown, and a recommended operating point) and never touches disk.

## How to run

```bash
uv sync --extra dev            # one-time: install deps
python -m benchmarks           # prints the report to stdout
```

It completes in ~1.5 s and needs no dependencies beyond the existing ones
(Pillow + numpy). Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--campaigns N` | all (6) | cap the number of scam campaigns |
| `--clean-count N` | 18 | number of benign negative images |
| `--steps N` | 25 | threshold-sweep granularity |
| `--hi F` | 0.30 | upper bound of the threshold sweep |
| `--candidate-radius N` | 12 | phash Hamming radius for BK-tree candidate gathering (production value) |
| `--markdown PATH` | – | also write the report to a Markdown file |
| `--json PATH` | – | write a machine-readable JSON artifact |

To regenerate the committed artifacts:

```bash
python -m benchmarks \
  --markdown docs/eval/detection-eval-report.md \
  --json docs/eval/detection-eval-report.json
```

## What the harness does

1. **Synthetic corpus** (`benchmarks/corpus.py`). For each scam *campaign* it
   renders a base image (banner + body text + QR-like block) and a family of
   deterministic re-share perturbations: `resize`, `crop`, `recompress` (JPEG
   q=35), `brightness`, `contrast`, `watermark` (text overlay), and `flip`
   (horizontal). Clean negatives are gradients, noise "photos", and bar charts.
   All seeds are fixed, so the corpus is byte-stable.
2. **Scoring** (`benchmarks/harness.py`). It builds a BK-tree
   `HashIndex` from the campaign *bases only*, then for every image gathers
   phash candidates within the candidate radius and keeps the lowest ensemble
   score (`optimus.hashing.ensemble.compare`). A lower score is a closer match.
3. **Threshold sweep & reporting** (`benchmarks/report.py`). It tallies a
   confusion matrix at each threshold, computes precision/recall/F1/FPR,
   evaluates the three shipped presets at their *ambiguous ceiling* (the
   matcher flags both SCAM and AMBIGUOUS verdicts), and recommends an operating
   point.

## How to read the results

- **Threshold** is the maximum ensemble score (weighted, normalized Hamming
  distance in `[0, 1]`) at which an image is flagged. Lower = stricter.
- **Recall** is the fraction of scam images (bases + perturbed re-shares) that
  are flagged. **FPR** is the fraction of clean images wrongly flagged.
- The **recommended operating point** is the highest-recall threshold that keeps
  **zero false positives** — the right trade for an auto-moderation action,
  where one wrongly-deleted benign image is far costlier than a missed re-share.
- The **per-perturbation table** shows which edit types survive matching. This is
  the most actionable view: it tells you *which* re-share transforms the pipeline
  is and isn't robust to.

## Results (full corpus, 2026-06-12)

Corpus: 6 campaigns, 48 scam images (6 bases + 42 variants), 18 clean negatives.
Full sweep and JSON in
[`docs/eval/detection-eval-report.md`](eval/detection-eval-report.md) /
[`.json`](eval/detection-eval-report.json).

**Recommended operating point:** score threshold `0.30`, precision **1.000**,
recall **0.792**, F1 **0.884**, FPR **0.000** (0 FP on 18 clean images).

### Shipped presets

| Preset | Ambig ceiling | TP | FP | TN | FN | Precision | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict | 0.240 | 38 | 0 | 18 | 10 | 1.000 | 0.792 | 0.000 |
| balanced | 0.170 | 37 | 0 | 18 | 11 | 1.000 | 0.771 | 0.000 |
| permissive | 0.120 | 36 | 0 | 18 | 12 | 1.000 | 0.750 | 0.000 |

### Threshold sweep (abridged)

| Threshold | Precision | Recall | FPR | F1 |
| --- | --- | --- | --- | --- |
| 0.036 | 1.000 | 0.417 | 0.000 | 0.588 |
| 0.072 | 1.000 | 0.604 | 0.000 | 0.753 |
| 0.120 | 1.000 | 0.750 | 0.000 | 0.857 |
| 0.144 | 1.000 | 0.771 | 0.000 | 0.871 |
| 0.216 | 1.000 | 0.792 | 0.000 | 0.884 |
| 0.300 | 1.000 | 0.792 | 0.000 | 0.884 |

### Per-perturbation recall (at the recommended threshold)

| Perturbation | Caught | Total | Recall |
| --- | --- | --- | --- |
| base | 6 | 6 | 1.000 |
| resize | 6 | 6 | 1.000 |
| recompress | 6 | 6 | 1.000 |
| brightness | 6 | 6 | 1.000 |
| contrast | 6 | 6 | 1.000 |
| watermark | 6 | 6 | 1.000 |
| crop | 2 | 6 | **0.333** |
| flip | 0 | 6 | **0.000** |

## Findings and known weaknesses

The headline result is reassuring: **precision and FPR are perfect (1.000 / 0.000)
across the entire threshold range**, even out to score 0.60 — clean uploads are
well-separated from the indexed scams, so the zero-false-positive guarantee the
auto-moderation action relies on holds with wide margin. The default ensemble
weights and preset thresholds (`optimus.hashing.ensemble`) are sound and need no
change. Resize, JPEG recompression, brightness/contrast shifts, and watermark
overlays are all caught at 100%.

Two perturbations are *not* caught, and the harness pins down exactly why:

1. **Horizontal flip — 0% recall (inherent, not a bug).** A flipped image has a
   phash Hamming distance of ~28–32 from its original. Perceptual hashes (aHash,
   dHash, pHash, wHash) are not flip-invariant by construction, and the pipeline
   has never claimed to be. Catching flips would require either indexing the
   mirrored hash of every known scam or a flip-invariant feature; both are out of
   scope for a hash-distance matcher and would be a design change, not a fix.
   This is documented here as an honest limitation rather than papered over.

2. **Heavy border crop — 33% recall, and the bottleneck is the candidate radius,
   not the threshold.** This is the most useful finding. Crop re-shares land at
   phash distance **12–16** from the base, straddling the production
   `DEFAULT_CANDIDATE_RADIUS = 12` (`optimus.services.detection.matcher`). Only
   the two crops at exactly distance 12 enter the BK-tree candidate set; the rest
   are *never scored by the ensemble at all*, so no threshold loosening can
   recover them — recall plateaus at 0.792 no matter how high the score
   threshold goes. Widening the candidate radius does recover them:

   | Candidate radius | crop recall | overall recall | precision | FPR |
   | --- | --- | --- | --- | --- |
   | 12 (production) | 0.333 | 0.792 | 1.000 | 0.000 |
   | 18 | **1.000** | **0.875** | 1.000 | 0.000 |

   On this corpus, radius 18 lifts crop recall from 33% to 100% and overall
   recall from 0.792 to 0.875 with **no loss of precision and zero false
   positives**. Verify with `python -m benchmarks --candidate-radius 18`.

### Why the candidate radius was left unchanged

The crop finding is a *tuning trade-off*, not a clear surgical bug, so this cycle
documents it rather than changing the production constant. The synthetic corpus
has only six well-separated campaigns; a wider candidate radius does more BK-tree
work per lookup and raises the chance of a real-world false candidate that the
ensemble then has to reject. Before raising `DEFAULT_CANDIDATE_RADIUS`,
maintainers should re-run this harness against a realistic index size and a
broader negative set to confirm the zero-FP property survives at scale. The knob
is exposed (`--candidate-radius`) precisely so that evaluation can happen offline
before any production change.

## Smoke test

`tests/unit/test_eval_harness.py` runs a tiny two-campaign corpus through the
full harness on every CI run, asserting determinism, that bases match themselves
exactly, that clean images are never flagged, that recall is monotonic in the
threshold, and the documented `flip`/`crop` behavior — so the eval stays runnable
and the findings above stay true.
