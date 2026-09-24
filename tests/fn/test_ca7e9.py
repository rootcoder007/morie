"""Tests for ca7e9.ca_chapter_7_equation_9."""

import math

from morie.fn import _array_core as np

from morie.fn.ca7e9 import ca_chapter_7_equation_9


def test_ca7e9_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b0 = 1.5
    bs = [0.5, -0.3, 0.2]
    xs = rng.normal(0, 1, len(bs))
    u_j = 0.7
    result = ca_chapter_7_equation_9(b0, bs, xs, u_j)
    assert isinstance(result, dict)
    assert "value" in result
    val = result["value"]
    if isinstance(val, (int, float)):
        assert math.isfinite(val)
    else:
        assert all(math.isfinite(v) for v in val)


def test_ca7e9_edge():
    """Test edge cases with small input."""
    rng = np.random.default_rng(43)
    b0 = 0.0
    bs = [0.0, 0.0]
    xs = rng.normal(0, 1, len(bs))
    u_j = 0.0
    result = ca_chapter_7_equation_9(b0, bs, xs, u_j)
    assert isinstance(result, dict)
    assert "value" in result
    val = result["value"]
    if isinstance(val, (int, float)):
        assert math.isfinite(val)
    else:
        assert all(math.isfinite(v) for v in val)
