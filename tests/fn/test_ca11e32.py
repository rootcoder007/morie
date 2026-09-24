"""Tests for ca11e32.ca_chapter_11_equation_32."""

import math

from morie.fn import _array_core as np

from morie.fn.ca11e32 import ca_chapter_11_equation_32


def test_ca11e32_basic():
    """Test basic functionality."""
    d = 0.5
    se_d = 0.1
    n1 = 50
    n2 = 60
    result = ca_chapter_11_equation_32(d, se_d, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_ca11e32_edge():
    """Test edge cases."""
    d = 0.2
    se_d = 0.05
    n1 = 5
    n2 = 7
    result = ca_chapter_11_equation_32(d, se_d, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
