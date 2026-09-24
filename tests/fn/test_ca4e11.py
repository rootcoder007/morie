"""Tests for ca4e11.ca_chapter_4_equation_11."""

import math

from morie.fn import _array_core as np

from morie.fn.ca4e11 import ca_chapter_4_equation_11


def test_ca4e11_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b = 0.5
    s = 0.25
    result = ca_chapter_4_equation_11(b, s)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isclose(result["value"], b * 2 * s)


def test_ca4e11_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    b = -1.5
    s = 0.3
    result = ca_chapter_4_equation_11(b, s)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert math.isclose(result["value"], b * 2 * s)
