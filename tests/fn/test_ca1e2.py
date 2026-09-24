"""Tests for ca1e2.ca_chapter_1_equation_2."""

import math

from morie.fn import _array_core as np

from morie.fn.ca1e2 import ca_chapter_1_equation_2


def test_ca1e2_basic():
    """Test basic functionality."""
    b0 = 0.5
    bs = [0.3, -0.2, 0.1, 0.4]
    xs = [1.0, -0.5, 2.0, 0.0]
    result = ca_chapter_1_equation_2(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
    assert math.isfinite(result["value"])
    # exp(.) is strictly positive
    assert result["value"] > 0


def test_ca1e2_edge():
    """Test edge cases."""
    # all-zero linear predictor -> exp(0) = 1
    b0 = 0.0
    bs = [0.0, 0.0, 0.0]
    xs = [0.0, 0.0, 0.0]
    result = ca_chapter_1_equation_2(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert abs(result["value"] - 1.0) < 1e-12
