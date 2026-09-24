"""Tests for ca4e5.ca_chapter_4_equation_5."""

from morie.fn import _array_core as np

from morie.fn.ca4e5 import ca_chapter_4_equation_5


def test_ca4e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b0 = 0.1
    bs = [0.5, -0.3, 0.2]
    xs = rng.normal(0, 1, 3)
    result = ca_chapter_4_equation_5(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    import math
    assert math.isfinite(result["value"])


def test_ca4e5_edge():
    """Test edge cases."""
    b0 = 0.0
    bs = [1.0]
    xs = [0.0]
    result = ca_chapter_4_equation_5(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    import math
    assert math.isfinite(result["value"])
    assert result["value"] == 0.0
