"""Tests for ca7e13.ca_chapter_7_equation_13."""

import math

from morie.fn import _array_core as np

from morie.fn.ca7e13 import ca_chapter_7_equation_13


def test_ca7e13_basic():
    """Test basic functionality."""
    b0 = 1.0
    b1 = 0.5
    x1 = 2.0
    u_0j = 0.1
    u_1j = 0.2
    result = ca_chapter_7_equation_13(b0, b1, x1, u_0j, u_1j)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_ca7e13_edge():
    """Test edge cases."""
    # All zeros should still yield a finite predicted value.
    result = ca_chapter_7_equation_13(0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
