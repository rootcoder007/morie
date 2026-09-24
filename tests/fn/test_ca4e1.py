"""Tests for ca4e1.ca_chapter_4_equation_1."""

from morie.fn import _array_core as np

from morie.fn.ca4e1 import ca_chapter_4_equation_1

import math


def test_ca4e1_basic():
    """Test basic functionality."""
    b0 = 0.5
    b1 = -1.2
    x1 = 0.3
    result = ca_chapter_4_equation_1(b0, b1, x1)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert "method" in result


def test_ca4e1_edge():
    """Test edge cases."""
    # Edge case: zero coefficient
    b0 = 1.0
    b1 = 0.0
    x1 = -2.0
    result = ca_chapter_4_equation_1(b0, b1, x1)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
