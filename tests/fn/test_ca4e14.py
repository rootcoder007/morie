"""Tests for ca4e14.ca_chapter_4_equation_14."""

from morie.fn import _array_core as np

from morie.fn.ca4e14 import ca_chapter_4_equation_14


def test_ca4e14_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    null = float(rng.normal(200, 10))
    full = float(rng.normal(150, 10))
    result = ca_chapter_4_equation_14(null, full)
    expected = null - full
    assert isinstance(result, dict)
    assert "value" in result
    assert abs(result["value"] - expected) < 1e-9


def test_ca4e14_edge():
    """Test edge cases."""
    null = 200.0
    full = 200.0
    result = ca_chapter_4_equation_14(null, full)
    expected = null - full
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == expected
    assert result["value"] == 0.0

    null2 = 300.0
    full2 = 250.0
    result2 = ca_chapter_4_equation_14(null2, full2)
    assert abs(result2["value"] - (null2 - full2)) < 1e-9
