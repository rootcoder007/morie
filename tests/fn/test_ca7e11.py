"""Tests for ca7e11.ca_chapter_7_equation_11."""

import math

from morie.fn.ca7e11 import ca_chapter_7_equation_11


def test_ca7e11_basic():
    """Test basic functionality."""
    b0, b1, b2 = 1.0, 0.5, 0.3
    x_ij, cluster_mean, u_j = 2.0, 1.5, 0.1
    result = ca_chapter_7_equation_11(b0, b1, b2, x_ij, cluster_mean, u_j)
    assert isinstance(result, dict)
    assert "value" in result
    expected = b0 + b1 * (x_ij - cluster_mean) + b2 * cluster_mean + u_j
    assert math.isclose(result["value"], expected)


def test_ca7e11_edge():
    """Test edge case with all zero parameters."""
    result = ca_chapter_7_equation_11(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] == 0.0
