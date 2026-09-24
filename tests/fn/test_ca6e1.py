"""Tests for ca6e1.ca_chapter_6_equation_1."""

import math

from morie.fn import _array_core as np

from morie.fn.ca6e1 import ca_chapter_6_equation_1


def test_ca6e1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b0 = 0.5
    b1 = 1.2
    x1 = rng.normal(0, 1, 1)[0]
    result = ca_chapter_6_equation_1(b0, b1, x1)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))


def test_ca6e1_edge():
    """Test edge cases."""
    b0 = 0.0
    b1 = 0.0
    x1 = 0.0
    result = ca_chapter_6_equation_1(b0, b1, x1)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))
