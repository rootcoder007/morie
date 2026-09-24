"""Tests for hmdeit.geron_deit."""

import math

from morie.fn import _array_core as np

from morie.fn.hmdeit import geron_deit


def test_hmdeit_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    image = rng.normal(0, 1, (3, 32, 32))
    result = geron_deit(image, patch_size=16, n_layers=1, d_model=8, n_heads=2, n_classes=4)
    assert isinstance(result, dict)
    assert result["n_patches"] == 4
    assert result["n_tokens"] == 6
    assert result["patch_embed_params"] == 6152
    assert result["distillation_overhead"] == 44
    assert math.isfinite(result["total_params"])
    assert math.isfinite(result["block_params"])
    assert "method" in result


def test_hmdeit_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    image = rng.normal(0, 1, (3, 16, 16))
    result = geron_deit(image, patch_size=16, n_layers=1, d_model=8, n_heads=2, n_classes=4)
    assert isinstance(result, dict)
    assert result["n_patches"] == 1
    assert result["n_tokens"] == 3
    assert math.isfinite(result["total_params"])
    assert math.isfinite(result["block_params"])
