"""Tests for ca6e3.ca_chapter_6_equation_3."""

from morie.fn import _array_core as np

from morie.fn.ca6e3 import ca_chapter_6_equation_3


def test_ca6e3_basic():
    """Test basic functionality with scalar inputs."""
    rng = np.random.default_rng(42)
    b0 = float(rng.normal(0, 1))
    b1 = float(rng.normal(0, 1))
    x1 = float(rng.normal(0, 1))
    result = ca_chapter_6_equation_3(b0, b1, x1)
    assert isinstance(result, dict)
    assert "value" in result
    expected = float(np.exp(b0 + b1 * x1))
    assert float(result["value"]) == expected


def test_ca6e3_edge():
    """Test edge case: zero inputs yield y = 1."""
    result = ca_chapter_6_equation_3(0.0, 0.0, 0.0)
    assert isinstance(result, dict)
    assert "value" in result
    expected = float(np.exp(0.0))
    assert float(result["value"]) == expected
