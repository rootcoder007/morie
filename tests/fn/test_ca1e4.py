"""Tests for ca1e4.ca_chapter_1_equation_4."""

import math

from morie.fn import _array_core as np

from morie.fn.ca1e4 import ca_chapter_1_equation_4


def test_ca1e4_basic():
    """Test basic functionality."""
    b0 = 0.5
    b1 = 1.2
    x1 = 0.7
    result = ca_chapter_1_equation_4(b0, b1, x1)
    assert isinstance(result, dict)
    assert "value" in result
    expected = b0 + b1 * x1
    assert math.isfinite(result["value"])
    assert math.isclose(result["value"], expected, rel_tol=1e-12)


def test_ca1e4_edge():
    """Test edge cases."""
    b0 = -0.3
    b1 = 0.0
    x1 = -1.5
    result = ca_chapter_1_equation_4(b0, b1, x1)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert math.isclose(result["value"], b0, rel_tol=1e-12)
    # Method label should also be present
    assert "method" in result
