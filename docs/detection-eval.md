# Detection-quality evaluation harness

The `benchmarks` package measures the offline detection quality of the
perceptual-hash + phash-index + ensemble pipeline: how well it re-identifies a known
scam image after realistic re-share edits (recall), how often it flags benign
uploads (false-positive rate), and how those trade off as the match threshold
moves. It runs the **real** detection code — the same hash functions, phash-index
candidate gathering (multi-index hashing), and ensemble scoring the bot uses in production — over a
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
| `--candidate-radius N` | 18 | phash Hamming radius for candidate gathering (production value) |
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
2. **Scoring** (`benchmarks/harness.py`). It builds a phash
   `HashIndex` (multi-index hashing) from the campaign *bases*, indexing each base together with its
   horizontal-flip *mirror* hash set (the same flip-invariant indexing the
   production pipeline performs — see below). For every image it then gathers
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
recall **1.000**, F1 **1.000**, FPR **0.000** (0 FP on 18 clean images).

### Shipped presets

| Preset | Ambig ceiling | TP | FP | TN | FN | Precision | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict | 0.240 | 48 | 0 | 18 | 0 | 1.000 | 1.000 | 0.000 |
| balanced | 0.170 | 47 | 0 | 18 | 1 | 1.000 | 0.979 | 0.000 |
| permissive | 0.120 | 43 | 0 | 18 | 5 | 1.000 | 0.896 | 0.000 |

### Threshold sweep (abridged)

| Threshold | Precision | Recall | FPR | F1 |
| --- | --- | --- | --- | --- |
| 0.036 | 1.000 | 0.542 | 0.000 | 0.703 |
| 0.072 | 1.000 | 0.729 | 0.000 | 0.843 |
| 0.120 | 1.000 | 0.896 | 0.000 | 0.945 |
| 0.168 | 1.000 | 0.979 | 0.000 | 0.989 |
| 0.216 | 1.000 | 1.000 | 0.000 | 1.000 |
| 0.300 | 1.000 | 1.000 | 0.000 | 1.000 |

### Per-perturbation recall (at the recommended threshold)

| Perturbation | Caught | Total | Recall |
| --- | --- | --- | --- |
| base | 6 | 6 | 1.000 |
| resize | 6 | 6 | 1.000 |
| recompress | 6 | 6 | 1.000 |
| brightness | 6 | 6 | 1.000 |
| contrast | 6 | 6 | 1.000 |
| watermark | 6 | 6 | 1.000 |
| crop | 6 | 6 | **1.000** |
| flip | 6 | 6 | **1.000** |

## Findings: the two weaknesses are now fixed

The headline result holds and improves: **precision and FPR remain perfect
(1.000 / 0.000) across the entire threshold range**, while overall recall rises
from **0.792 to 1.000** at the zero-FP operating point. The default ensemble
weights and preset thresholds (`optimus.hashing.ensemble`) are unchanged — both
fixes are upstream of scoring (what gets indexed and how candidates are
gathered), so the zero-false-positive guarantee the auto-moderation action relies
on is untouched. Resize, JPEG recompression, brightness/contrast shifts, and
watermark overlays were already caught at 100%; this cycle closes the last two
gaps the harness had pinned down.

1. **Horizontal flip — now 100% recall (flip-invariant indexing).** A flipped
   image has a phash Hamming distance of ~28–32 from its original, and perceptual
   hashes (aHash, dHash, pHash, wHash) are not flip-invariant by construction. The
   fix is to index the mirror up front: when a scam image is added,
   `compute_all_mirror` hashes the horizontally-flipped pixels and those four
   hashes are stored (`mphash`/`mdhash`/`mwhash`/`mahash`, nullable — migration
   `0004`) and indexed as a **sibling index entry under the same `hash_id` and
   `campaign_id`** (`optimus.services.detection.index`). A mirrored re-share then
   matches the mirror entry at ~zero ensemble distance, and because the sibling
   carries the source `hash_id`, the match resolves back to the *same* source
   detection — dedup and ownership semantics are preserved. The mirror is derived
   from the actual flipped pixels (not a bit-permutation of the original hash,
   which the area-resize and DCT/median do not preserve), so the match is exact.
   Rows added before this change, or via the typed-hex `/scamhash add` path that
   never sees an image, leave the mirror columns NULL and simply contribute no
   mirror entry.

2. **Heavy border crop — now 100% recall (candidate radius 12 → 18).** Crop
   re-shares land at phash distance **12–16** from the base, straddling the old
   `DEFAULT_CANDIDATE_RADIUS = 12` (`optimus.services.detection.matcher`): only
   the crops at exactly distance 12 entered the candidate set, so recall plateaued
   at 0.792 no matter how high the score threshold went (the rest were never
   scored). Raising the radius to **18** lets every crop variant enter the
   candidate set, where the ensemble scores them well within threshold.

The two fixes are independent and compose cleanly:

| Configuration | crop recall | flip recall | overall recall | precision | FPR |
| --- | --- | --- | --- | --- | --- |
| baseline (radius 12, no mirror) | 0.333 | 0.000 | 0.792 | 1.000 | 0.000 |
| + flip-invariant indexing (radius 12) | 0.333 | **1.000** | 0.917 | 1.000 | 0.000 |
| + candidate radius 18 (shipped) | **1.000** | **1.000** | **1.000** | 1.000 | 0.000 |

Verify the intermediate row with `python -m benchmarks --candidate-radius 12`.

### On the candidate radius trade-off

Radius 18 does marginally more index work per lookup and, on a larger and
noisier real-world index, could surface more false *candidates* for the ensemble
to reject — but on this corpus precision and FPR stay at 1.000 / 0.000 with wide
margin, and the ensemble (not the radius) is what decides a match. The knob stays
exposed (`--candidate-radius`) so maintainers can re-run this harness against a
realistic index size and a broader negative set before tuning further.

## Smoke test

`tests/unit/test_eval_harness.py` runs a tiny two-campaign corpus through the
full harness on every CI run, asserting determinism, that bases match themselves
exactly, that clean images are never flagged, that recall is monotonic in the
threshold, and the documented `flip`/`crop` behavior (both now caught) — so the
eval stays runnable and the findings above stay true.
