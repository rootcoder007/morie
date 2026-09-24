"""Tests for clipxi.clip_image_encoder."""

import math

from morie.fn import _array_core as np

from morie.fn.clipxi import clip_image_encoder


def test_clipxi_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # vit-l/14 has patch size 14; image must have dims that are multiples of 14
    image = rng.normal(0, 1, (14, 14))
    result = clip_image_encoder(image, backbone="vit-l/14")
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "embedding" in result
    assert "n_patches" in result
    assert "grid" in result
    assert "patch" in result
    assert "width" in result
    assert "embed_dim" in result
    assert "norm" in result
    assert math.isfinite(result["estimate"])


def test_clipxi_edge():
    """Test edge cases."""
    # vit-b/32 has patch size 32; use a different backbone and seed
    rng = np.random.default_rng(7)
    image = rng.normal(0, 1, (32, 32))
    result = clip_image_encoder(image, backbone="vit-b/32", seed=7)
    assert isinstance(result, dict)
    assert "embedding" in result
    assert "n_patches" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["norm"])
    # The projected embedding has length out_dim = min(cfg["embed"], 32) = 32,
    # while embed_dim reports the full backbone embedding dimension.
    assert len(result["embedding"]) == 32
