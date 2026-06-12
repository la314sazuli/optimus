"""Offline detection-quality evaluation harness for the scam-image pipeline.

This package builds a deterministic synthetic image corpus, runs the *real*
detection code (perceptual hashes -> phash index (MIH) -> ensemble vote) over it at
a sweep of match thresholds, and reports precision/recall/F1, per-perturbation
recall, and a recommended operating point. See ``docs/detection-eval.md``.
"""

from __future__ import annotations
