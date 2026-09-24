"""Tests for ca2e6.ca_chapter_2_equation_6."""

import math

from morie.fn import _array_core as np

from morie.fn.ca2e6 import ca_chapter_2_equation_6


def test_ca2e6_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    y = rng.normal(0, 1, 40)
    result = ca_chapter_2_equation_6(x, y)
    assert isinstance(result, dict)
    assert "t_from_r" in result
    assert math.isfinite(result["t_from_r"])


def test_ca2e6_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    y = x + rng.normal(0, 0.1, 40)
    result = ca_chapter_2_equation_6(x, y)
    assert isinstance(result, dict)
    assert "t_from_r" in result
    assert math.isfinite(result["t_from_r"])
