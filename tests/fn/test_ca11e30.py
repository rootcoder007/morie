"""Tests for ca11e30.ca_chapter_11_equation_30."""

from morie.fn import _array_core as np

from morie.fn.ca11e30 import ca_chapter_11_equation_30


def test_ca11e30_basic():
    """Test basic functionality."""
    d = 0.5
    n1 = 30
    n2 = 50
    expected = d / np.sqrt(d ** 2 + (n1 + n2) ** 2 / (n1 * n2))
    result = ca_chapter_11_equation_30(d, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result
    assert np.isclose(result["value"], expected)


def test_ca11e30_edge():
    """Test edge cases."""
    d = 0.0
    n1 = 10
    n2 = 10
    result = ca_chapter_11_equation_30(d, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == 0.0
