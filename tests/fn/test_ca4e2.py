"""Tests for ca4e2.ca_chapter_4_equation_2."""

import math

from morie.fn import _array_core as np

from morie.fn.ca4e2 import ca_chapter_4_equation_2


def test_ca4e2_basic():
    """Test basic functionality."""
    b0 = 0.5
    b1 = -1.2
    x1 = 0.3
    result = ca_chapter_4_equation_2(b0, b1, x1)
    assert isinstance(result, dict)
    assert "value" in result
    # Independent computation of the formula: P(Y=1) = 1 / (1 + e^-(b0 + b1*x1))
    expected = 1.0 / (1.0 + math.exp(-(b0 + b1 * x1)))
    assert abs(result["value"] - expected) < 1e-12


def test_ca4e2_edge():
    """Test edge cases."""
    # Edge case: zero linear predictor -> probability should be 0.5
    b0 = 0.0
    b1 = 0.0
    x1 = 0.0
    result = ca_chapter_4_equation_2(b0, b1, x1)
    assert isinstance(result, dict)
    assert abs(result["value"] - 0.5) < 1e-12
    expected = 1.0 / (1.0 + math.exp(-(b0 + b1 * x1)))
    assert abs(result["value"] - expected) < 1e-12
