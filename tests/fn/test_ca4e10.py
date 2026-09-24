"""Tests for ca4e10.ca_chapter_4_equation_10."""

from morie.fn import _array_core as np

import math

from morie.fn.ca4e10 import ca_chapter_4_equation_10


def test_ca4e10_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b = rng.normal(0, 1)
    s = rng.uniform(0.1, 2.0)
    result = ca_chapter_4_equation_10(b, s)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_ca4e10_edge():
    """Test edge cases."""
    result = ca_chapter_4_equation_10(0.0, 1.0)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] == 0.0
