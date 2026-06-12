"""Deterministic synthetic image corpus for the detection eval harness.

Generates, fully in memory, a set of scam "campaign" base images plus a family
of realistic re-share perturbations (resize, crop, JPEG recompress, brightness
and contrast shifts, watermark/text overlay, horizontal flip) and a pool of
clean negatives (gradients, noise "photos", bar charts). Everything is keyed off
fixed seeds so the corpus is byte-stable across runs and machines.

The corpus mirrors the on-disk fixtures in ``tests/fixtures`` but is richer (more
perturbation kinds, scalable size) and never touches disk, which keeps the
harness fast and dependency-free beyond Pillow/numpy.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

SIZE = 256

# Each campaign is a base scam image; the harness indexes the base and expects
# every perturbation of it to still match. (color, title, subtitle.)
_CAMPAIGNS: tuple[tuple[str, tuple[int, int, int], str, str], ...] = (
    ("nitro_gift", (88, 101, 242), "FREE NITRO", "CLAIM NOW"),
    ("steam_gift", (27, 40, 56), "STEAM GIFT", "50$ CARD"),
    ("crypto_drain", (240, 185, 11), "DOUBLE BTC", "SCAN QR"),
    ("giveaway", (237, 66, 69), "MEGA GIVEAWAY", "WINNER!"),
    ("airdrop", (16, 124, 16), "TOKEN AIRDROP", "CONNECT WALLET"),
    ("support_dm", (114, 137, 218), "OFFICIAL SUPPORT", "VERIFY HERE"),
)

# Ordered so the harness can report a stable per-perturbation table.
PERTURBATIONS: tuple[str, ...] = (
    "resize",
    "crop",
    "recompress",
    "brightness",
    "contrast",
    "watermark",
    "flip",
)


@dataclass(frozen=True, slots=True)
class CorpusImage:
    """One labeled corpus image held in memory.

    ``campaign`` is the scam campaign name for scam images (``None`` for clean
    negatives). ``perturbation`` is ``"base"`` for an unmodified base image, the
    perturbation name for a variant, or ``None`` for clean negatives.
    """

    name: str
    image: Image.Image
    is_scam: bool
    campaign: str | None
    perturbation: str | None

    @property
    def is_base(self) -> bool:
        """Whether this is an indexed campaign base (not a re-share variant)."""
        return self.perturbation == "base"


@dataclass(frozen=True, slots=True)
class Corpus:
    """A generated corpus: campaign bases, scam variants, and clean negatives."""

    bases: tuple[CorpusImage, ...]
    variants: tuple[CorpusImage, ...]
    cleans: tuple[CorpusImage, ...]

    def scam_images(self) -> Iterator[CorpusImage]:
        """Yield every scam image (bases first, then perturbed variants)."""
        yield from self.bases
        yield from self.variants

    def all_images(self) -> Iterator[CorpusImage]:
        """Yield every image in the corpus."""
        yield from self.bases
        yield from self.variants
        yield from self.cleans

    @property
    def num_campaigns(self) -> int:
        """Number of distinct scam campaigns."""
        return len(self.bases)


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
    """Render one scam base image (banner + body text + QR-like block)."""
    img = Image.new("RGB", (SIZE, SIZE), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((8, 8, SIZE - 8, 64), fill=(0, 0, 0))
    draw.text((20, 24), title, fill=(255, 255, 255))
    draw.text((20, 80), sub, fill=(255, 255, 255))
    # Stable per-campaign seed (independent of Python's salted hash()).
    seed = int.from_bytes(name.encode("utf-8"), "little") % (2**32)
    _draw_qr_like(draw, seed, (SIZE - 120, SIZE - 120, SIZE - 16, SIZE - 16))
    return img


def _perturb(base: Image.Image, kind: str) -> Image.Image:
    """Apply a named, deterministic perturbation modeling a re-shared scam."""
    if kind == "resize":
        small = base.resize((180, 180), Image.Resampling.LANCZOS)
        return small.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    if kind == "crop":
        cropped = base.crop((10, 10, SIZE - 10, SIZE - 10))
        return cropped.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    if kind == "recompress":
        buf = io.BytesIO()
        base.save(buf, format="JPEG", quality=35)
        buf.seek(0)
        with Image.open(buf) as reopened:
            return reopened.convert("RGB")
    if kind == "brightness":
        return ImageEnhance.Brightness(base).enhance(1.25)
    if kind == "contrast":
        return ImageEnhance.Contrast(base).enhance(0.8)
    if kind == "watermark":
        wm = base.copy()
        draw = ImageDraw.Draw(wm)
        draw.text((40, SIZE // 2), "@discord", fill=(255, 255, 255))
        return wm
    if kind == "flip":
        return base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    raise ValueError(f"unknown perturbation: {kind}")


def _clean_images(count: int, seed: int) -> list[CorpusImage]:
    """Build ``count`` deterministic benign images (gradients/noise/charts)."""
    out: list[CorpusImage] = []
    rng = np.random.default_rng(seed)
    kinds = ("gradient", "photo_noise", "chart")
    for i in range(count):
        kind = kinds[i % len(kinds)]
        if kind == "gradient":
            ramp = np.linspace(0, 255, SIZE, dtype=np.float64)
            arr = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
            arr[:, :, i % 3] = ramp[None, :]
            arr[:, :, (i + 1) % 3] = ramp[:, None]
            img = Image.fromarray(arr.astype(np.uint8), "RGB")
        elif kind == "photo_noise":
            noise = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
            img = Image.fromarray(noise, "RGB")
        else:
            img = Image.new("RGB", (SIZE, SIZE), (245, 245, 245))
            draw = ImageDraw.Draw(img)
            heights = rng.integers(40, SIZE - 20, size=8)
            for b, h in enumerate(heights):
                x0 = 16 + b * 28
                draw.rectangle((x0, SIZE - int(h), x0 + 20, SIZE - 16), fill=(60, 120, 200))
        out.append(
            CorpusImage(
                name=f"{kind}_{i}",
                image=img,
                is_scam=False,
                campaign=None,
                perturbation=None,
            )
        )
    return out


def build_corpus(
    *,
    campaigns: int | None = None,
    perturbations: tuple[str, ...] = PERTURBATIONS,
    clean_count: int = 18,
    clean_seed: int = 2024,
) -> Corpus:
    """Build the synthetic corpus.

    ``campaigns`` caps the number of scam campaigns (``None`` uses all built-in
    ones); ``perturbations`` selects which re-share variants to generate per
    campaign; ``clean_count`` sets the number of benign negatives. The result is
    deterministic for a given set of arguments.
    """
    selected = _CAMPAIGNS if campaigns is None else _CAMPAIGNS[:campaigns]
    bases: list[CorpusImage] = []
    variants: list[CorpusImage] = []
    for name, color, title, sub in selected:
        base_img = _scam_base(name, color, title, sub)
        bases.append(
            CorpusImage(
                name=name,
                image=base_img,
                is_scam=True,
                campaign=name,
                perturbation="base",
            )
        )
        for kind in perturbations:
            variants.append(
                CorpusImage(
                    name=f"{name}_{kind}",
                    image=_perturb(base_img, kind),
                    is_scam=True,
                    campaign=name,
                    perturbation=kind,
                )
            )
    cleans = tuple(_clean_images(clean_count, clean_seed))
    return Corpus(bases=tuple(bases), variants=tuple(variants), cleans=cleans)
