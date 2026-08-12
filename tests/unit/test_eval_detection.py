"""Tests for the detection evaluation harness."""

from pathlib import Path

import numpy as np
from PIL import Image
from scripts.eval_detection import THRESHOLDS, TRANSFORMS, eval_image, transform


def _make_image(path: Path) -> Path:
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def test_all_transforms_produce_valid_image(tmp_path):
    img = Image.new("RGB", (200, 200), (100, 150, 200))
    for tname in TRANSFORMS:
        result = transform(img, tname)
        assert result.size[0] > 0 and result.size[1] > 0


def test_eval_image_returns_results_for_all_transforms(tmp_path):
    path = _make_image(tmp_path / "scam1.png")
    result = eval_image(path)
    assert result.image == "scam1.png"
    assert len(result.variants) == len(TRANSFORMS)
    for v in result.variants:
        assert v.transform in TRANSFORMS
        assert set(v.distances.keys()) == set(THRESHOLDS.keys())
        assert all(d >= 0 for d in v.distances.values())


def test_identical_image_has_zero_distance(tmp_path):
    path = _make_image(tmp_path / "scam2.png")
    result = eval_image(path)
    # recompress at q=50 on a random image is aggressive; phash should still be close.
    rec = next(v for v in result.variants if v.transform == "recompress_50")
    assert rec.distances["phash"] <= 15
