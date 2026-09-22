"""Tests for ca4e3.ca_chapter_4_equation_3."""

from morie.fn import _array_core as np

from morie.fn.ca4e3 import ca_chapter_4_equation_3


def test_ca4e3_basic():
    """Test basic functionality with a scalar input."""
    rng = np.random.default_rng(42)
    x = float(rng.normal(0, 1))
    result = ca_chapter_4_equation_3(x)
    assert isinstance(result, dict)
    assert "value" in result
    # Compare against the documented formula: P = 1 / (1 + e^-x)
    expected = 1.0 / (1.0 + np.exp(-x))
    assert result["value"] == expected


def test_ca4e3_zero():
    """Test that xb=0 yields 1/2 per the logistic formula."""
    result = ca_chapter_4_equation_3(0.0)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == 1.0 / (1.0 + np.exp(0.0))


def test_ca4e3_edge():
    """Test edge cases with a scalar input."""
    x = float(np.random.default_rng(42).normal(0, 1))
    result = ca_chapter_4_equation_3(x)
    assert isinstance(result, dict)
    assert "value" in result
