"""Tests for embedding helpers that do not require an ONNX model."""

from __future__ import annotations

import numpy as np
import pytest

from optimus.core.config import get_settings
from optimus.hashing import embedding
from optimus.hashing.embedding import EmbeddingUnavailableError


def test_cosine_similarity_basic() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert embedding.cosine_similarity(a, b) == 1.0
    assert embedding.cosine_similarity(a, c) == 0.0


def test_cosine_similarity_zero_vector() -> None:
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert embedding.cosine_similarity(a, b) == 0.0


def test_is_enabled_reflects_settings(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    get_settings.cache_clear()
    monkeypatch.setenv("OPTIMUS_EMBEDDING_ENABLED", "false")
    assert embedding.is_enabled() is False
    get_settings.cache_clear()
    monkeypatch.setenv("OPTIMUS_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("OPTIMUS_EMBEDDING_MODEL_PATH", "/tmp/model.onnx")
    assert embedding.is_enabled() is True
    get_settings.cache_clear()


def test_load_session_raises_when_disabled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    get_settings.cache_clear()
    embedding._load_session.cache_clear()
    monkeypatch.setenv("OPTIMUS_EMBEDDING_ENABLED", "false")
    with pytest.raises(EmbeddingUnavailableError, match="disabled"):
        embedding._load_session()
    embedding._load_session.cache_clear()
    get_settings.cache_clear()


def test_load_session_raises_when_model_path_unset(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    get_settings.cache_clear()
    embedding._load_session.cache_clear()
    monkeypatch.setenv("OPTIMUS_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("OPTIMUS_EMBEDDING_MODEL_PATH", "")
    with pytest.raises(EmbeddingUnavailableError, match="MODEL_PATH"):
        embedding._load_session()
    embedding._load_session.cache_clear()
    get_settings.cache_clear()


def test_preprocess_shapes_and_normalizes() -> None:
    gray = np.full((128, 96), 255.0)
    out = embedding._preprocess(gray, size=64)
    assert out.shape == (1, 1, 64, 64)
    assert out.dtype == np.float32
    # 255 -> 1.0 after the /255 normalization in _preprocess.
    assert float(out.max()) == pytest.approx(1.0)


def test_embed_l2_normalizes_session_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _FakeInput:
        name = "x"

    class _FakeSession:
        def get_inputs(self) -> list[_FakeInput]:
            return [_FakeInput()]

        def run(self, _outs: object, inputs: dict[str, object]) -> list[np.ndarray]:
            assert "x" in inputs  # preprocessed tensor keyed by the input name
            return [np.array([[3.0, 4.0]])]  # norm 5 -> expect [0.6, 0.8]

    monkeypatch.setattr(embedding, "_load_session", lambda: _FakeSession())
    vec = embedding.embed(np.zeros((64, 64)))
    assert vec.shape == (2,)
    assert np.allclose(vec, [0.6, 0.8])
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0)
