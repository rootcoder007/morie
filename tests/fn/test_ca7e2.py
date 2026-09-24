"""Tests for ca7e2.ca_chapter_7_equation_2."""

import math

from morie.fn import _array_core as np

from morie.fn.ca7e2 import ca_chapter_7_equation_2


def test_ca7e2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b0 = 1.5
    bs = [0.5, -0.3, 0.8]
    xs = rng.normal(0, 1, 3)
    result = ca_chapter_7_equation_2(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_ca7e2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    b0 = 0.0
    bs = [1.0]
    xs = rng.normal(0, 1, 1)
    result = ca_chapter_7_equation_2(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
