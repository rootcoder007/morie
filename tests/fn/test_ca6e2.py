"""Tests for ca6e2.ca_chapter_6_equation_2."""

import math

from morie.fn import _array_core as np

from morie.fn.ca6e2 import ca_chapter_6_equation_2


def test_ca6e2_basic():
    """Test basic functionality with scalar inputs."""
    beta0 = 0.5
    beta1 = 1.2
    x1 = 2.0
    result = ca_chapter_6_equation_2(beta0, beta1, x1)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))


def test_ca6e2_edge():
    """Test edge case with all-zero inputs."""
    result = ca_chapter_6_equation_2(0.0, 0.0, 0.0)
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
    assert math.isfinite(float(result["value"]))
