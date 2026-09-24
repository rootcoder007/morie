"""Tests for opnclp.open_clip."""

import math

from morie.fn import _array_core as np

from morie.fn.opnclp import open_clip


def test_opnclp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.1, 1.0, 100)
    texts = rng.uniform(0.1, 1.0, 100)
    result = open_clip(images, texts)
    assert isinstance(result, dict)
    assert len(result) > 0
    for value in result.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)


def test_opnclp_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.1, 1.0, 10)
    texts = rng.uniform(0.1, 1.0, 10)
    result = open_clip(images, texts)
    assert isinstance(result, dict)
    assert len(result) > 0
