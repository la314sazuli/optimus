"""Deterministically generate a small labeled image dataset for evaluation.

Produces synthetic "fake giveaway"-style scam images (bold text banners over a
colored field with a QR-like block) plus perturbed variants of each base
(resize, crop, recompress, recolor, watermark) — these model the same scam
re-shared after light edits, so a good perceptual matcher should still catch
them. Clean images (gradients, noise "photos", bar charts) model benign uploads.

Everything is seeded so re-running yields byte-stable files. Output layout::

    tests/fixtures/scam/<name>.<ext>
    tests/fixtures/clean/<name>.<ext>
    tests/fixtures/labels.json   # {"scam": [...], "clean": [...], "bases": {...}}

Run with: ``python scripts/make_fixtures.py``.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SIZE = 256

# A handful of distinct scam "campaigns", each a base image + perturbations.
_SCAM_CAMPAIGNS = (
    ("nitro_gift", (88, 101, 242), "FREE NITRO", "CLAIM NOW"),
    ("steam_gift", (27, 40, 56), "STEAM GIFT", "50$ CARD"),
    ("crypto_drain", (240, 185, 11), "DOUBLE BTC", "SCAN QR"),
    ("giveaway", (237, 66, 69), "MEGA GIVEAWAY", "WINNER!"),
)


@dataclass
class Manifest:
    """The labeled-file manifest written alongside the fixtures."""

    scam: list[str] = field(default_factory=list)
    clean: list[str] = field(default_factory=list)
    bases: dict[str, str] = field(default_factory=dict)


def _draw_qr_like(draw: ImageDraw.ImageDraw, seed: int, box: tuple[int, int, int, int]) -> None:
    """Draw a deterministic QR-code-like block of cells in ``box``."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    cells = 12
    cw = (x1 - x0) // cells
    ch = (y1 - y0) // cells
    draw.rectangle(box, fill=(255, 255, 255))
    for r in range(cells):
        for c in range(cells):
            if rng.random() < 0.5:
                cx0 = x0 + c * cw
                cy0 = y0 + r * ch
                draw.rectangle((cx0, cy0, cx0 + cw, cy0 + ch), fill=(0, 0, 0))


def _scam_base(name: str, color: tuple[int, int, int], title: str, sub: str) -> Image.Image:
    """Render one scam base image."""
    img = Image.new("RGB", (SIZE, SIZE), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((8, 8, SIZE - 8, 64), fill=(0, 0, 0))
    draw.text((20, 24), title, fill=(255, 255, 255))
    draw.text((20, 80), sub, fill=(255, 255, 255))
    seed = abs(hash(name)) % (2**32)
    _draw_qr_like(draw, seed, (SIZE - 120, SIZE - 120, SIZE - 16, SIZE - 16))
    return img


def _perturb(base: Image.Image, kind: str) -> Image.Image:
    """Apply a named, deterministic perturbation modeling a re-shared scam."""
    if kind == "resize":
        small = base.resize((180, 180), Image.LANCZOS)
        return small.resize((SIZE, SIZE), Image.LANCZOS)
    if kind == "crop":
        cropped = base.crop((10, 10, SIZE - 10, SIZE - 10))
        return cropped.resize((SIZE, SIZE), Image.LANCZOS)
    if kind == "recolor":
        arr = np.asarray(base, dtype=np.int16)
        arr = np.clip(arr + np.array([12, -8, 6], dtype=np.int16), 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGB")
    if kind == "watermark":
        wm = base.copy()
        draw = ImageDraw.Draw(wm)
        draw.text((40, SIZE // 2), "@discord", fill=(255, 255, 255))
        return wm
    if kind == "recompress":
        buf = io.BytesIO()
        base.save(buf, format="JPEG", quality=35)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(f"unknown perturbation: {kind}")


def _clean_images() -> list[tuple[str, Image.Image]]:
    """Build a deterministic set of benign images."""
    out: list[tuple[str, Image.Image]] = []
    rng = np.random.default_rng(2024)

    for i in range(4):
        base = np.linspace(0, 255, SIZE, dtype=np.float64)
        arr = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
        arr[:, :, i % 3] = base[None, :]
        arr[:, :, (i + 1) % 3] = base[:, None]
        out.append((f"gradient_{i}", Image.fromarray(arr.astype(np.uint8), "RGB")))

    for i in range(4):
        noise = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
        out.append((f"photo_noise_{i}", Image.fromarray(noise, "RGB")))

    for i in range(4):
        img = Image.new("RGB", (SIZE, SIZE), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        heights = rng.integers(40, SIZE - 20, size=8)
        for b, h in enumerate(heights):
            x0 = 16 + b * 28
            draw.rectangle((x0, SIZE - int(h), x0 + 20, SIZE - 16), fill=(60, 120, 200))
        out.append((f"chart_{i}", img))

    return out


def _save(img: Image.Image, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "JPEG":
        img.save(path, format="JPEG", quality=90)
    else:
        img.save(path, format=fmt)


def generate() -> Manifest:
    """Generate the full fixture set and return its manifest."""
    manifest = Manifest()
    scam_dir = FIXTURES / "scam"
    clean_dir = FIXTURES / "clean"

    perturbations = ("resize", "crop", "recolor", "watermark", "recompress")
    for name, color, title, sub in _SCAM_CAMPAIGNS:
        base = _scam_base(name, color, title, sub)
        base_path = scam_dir / f"{name}.png"
        _save(base, base_path, "PNG")
        manifest.scam.append(str(base_path.relative_to(FIXTURES)))
        manifest.bases[str(base_path.relative_to(FIXTURES))] = name
        for kind in perturbations:
            variant = _perturb(base, kind)
            ext = "jpg" if kind == "recompress" else "png"
            fmt = "JPEG" if ext == "jpg" else "PNG"
            vpath = scam_dir / f"{name}_{kind}.{ext}"
            _save(variant, vpath, fmt)
            manifest.scam.append(str(vpath.relative_to(FIXTURES)))
            manifest.bases[str(vpath.relative_to(FIXTURES))] = name

    for name, img in _clean_images():
        cpath = clean_dir / f"{name}.png"
        _save(img, cpath, "PNG")
        manifest.clean.append(str(cpath.relative_to(FIXTURES)))

    (FIXTURES / "labels.json").write_text(
        json.dumps(
            {"scam": manifest.scam, "clean": manifest.clean, "bases": manifest.bases},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest


def main() -> None:
    """Entry point: generate fixtures and print a summary."""
    manifest = generate()
    print(f"scam images:  {len(manifest.scam)}")
    print(f"clean images: {len(manifest.clean)}")
    print(f"total:        {len(manifest.scam) + len(manifest.clean)}")


if __name__ == "__main__":
    main()
