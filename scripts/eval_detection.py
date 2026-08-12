#!/usr/bin/env python
"""Detection evaluation harness.

Tests how well perceptual hashes survive image transformations.
Generates variants (crop, resize, recompress, rotate, color shift, watermark)
of known scam images and measures hamming distance to the original hash.

Usage:
    python scripts/eval_detection.py --images-dir ./fixtures/scam_images
    python scripts/eval_detection.py --images-dir ./fixtures/scam_images --json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from optimus.hashing.perceptual import compute_all, compute_all_mirror, hamming

TRANSFORMS = [
    "crop_5pct",
    "crop_15pct",
    "resize_50pct",
    "resize_150pct",
    "recompress_50",
    "recompress_10",
    "rotate_5",
    "color_shift",
    "watermark",
    "mirror",
]

THRESHOLDS = {"ahash": 10, "dhash": 10, "phash": 12, "whash": 10}


@dataclass
class VariantResult:
    transform: str
    distances: dict[str, int]
    matched: bool


@dataclass
class ImageResult:
    image: str
    variants: list[VariantResult] = field(default_factory=list)


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    w = np.array([0.299, 0.587, 0.114])
    return arr.astype(np.float64) @ w


def to_gray(rgb: Image.Image) -> np.ndarray:
    arr = np.array(rgb.convert("RGB"))
    w = np.array([0.299, 0.587, 0.114])
    return arr.astype(np.float64) @ w


def transform(img: Image.Image, name: str) -> Image.Image:
    w, h = img.size
    if name == "crop_5pct":
        return img.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
    if name == "crop_15pct":
        return img.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)))
    if name == "resize_50pct":
        return img.resize((int(w * 0.5), int(h * 0.5)))
    if name == "resize_150pct":
        return img.resize((int(w * 1.5), int(h * 1.5)))
    if name == "recompress_50":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=50)
        return Image.open(buf)
    if name == "recompress_10":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=10)
        return Image.open(buf)
    if name == "rotate_5":
        return img.rotate(5, expand=True, fillcolor=(128, 128, 128))
    if name == "color_shift":
        arr = np.array(img.convert("RGB")).astype(np.int16)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + 30, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - 30, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    if name == "watermark":
        out = img.convert("RGB").copy()
        px = out.load()
        for y in range(0, h, 3):
            for x in range(0, w, 3):
                r, g, b = px[x, y]
                px[x, y] = (min(r + 40, 255), min(g + 40, 255), min(b + 40, 255))
        return out
    if name == "mirror":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    raise ValueError(f"unknown transform: {name}")


def eval_image(path: Path) -> ImageResult:
    img = Image.open(path).convert("RGB")
    gray = to_gray(img)
    orig = compute_all(gray)
    if "mirror" in TRANSFORMS:
        orig_mirror = compute_all_mirror(gray)

    result = ImageResult(image=path.name)
    for tname in TRANSFORMS:
        variant = transform(img, tname)
        vgray = to_gray(variant)
        vhashes = compute_all(vgray) if tname != "mirror" else orig_mirror

        distances = {}
        for algo in orig:
            distances[algo] = hamming(orig[algo], vhashes[algo])
        matched = any(d <= THRESHOLDS[a] for a, d in distances.items())
        result.variants.append(VariantResult(tname, distances, matched))
    return result


def print_report(results: list[ImageResult]) -> None:
    print(f"{'Transform':<18} {'TP':>4} {'FN':>4} {'Rate':>6}  "
          f"{'ahash':>6} {'dhash':>6} {'phash':>6} {'whash':>6}")
    print("-" * 68)
    for tname in TRANSFORMS:
        total = 0
        tp = 0
        sums = dict.fromkeys(THRESHOLDS, 0)
        for r in results:
            for v in r.variants:
                if v.transform == tname:
                    total += 1
                    if v.matched:
                        tp += 1
                    for a, d in v.distances.items():
                        sums[a] += d
        fn = total - tp
        rate = tp / total * 100 if total else 0
        avgs = {a: sums[a] / total if total else 0 for a in THRESHOLDS}
        print(f"{tname:<18} {tp:>4} {fn:>4} {rate:>5.1f}%  "
              f"{avgs['ahash']:>6.1f} {avgs['dhash']:>6.1f} "
              f"{avgs['phash']:>6.1f} {avgs['whash']:>6.1f}")

    total_variants = sum(len(r.variants) for r in results)
    total_tp = sum(1 for r in results for v in r.variants if v.matched)
    print("-" * 68)
    print(f"{'TOTAL':<18} {total_tp:>4} {total_variants - total_tp:>4} "
          f"{total_tp / total_variants * 100 if total_variants else 0:>5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, required=True,
                    help="Directory of known scam images")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    args = ap.parse_args()

    images = sorted(
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif")
    )
    if not images:
        print(f"No images found in {args.images_dir}", file=sys.stderr)
        return 1

    results = [eval_image(p) for p in images]

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print(f"\nEvaluated {len(results)} images x {len(TRANSFORMS)} transforms\n")
        print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
