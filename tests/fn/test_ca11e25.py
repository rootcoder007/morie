"""Tests for ca11e25.ca_chapter_11_equation_25."""

from morie.fn import _array_core as np

from morie.fn.ca11e25 import ca_chapter_11_equation_25


def test_ca11e25_basic():
    """Test basic functionality with a single positive se_d value."""
    se_d = 0.5
    result = ca_chapter_11_equation_25(se_d)
    assert isinstance(result, dict)
    assert "value" in result
    # Formula: se_lnOR = sqrt(se_d^2 / 0.551^2)
    expected = np.sqrt(se_d ** 2 / 0.551 ** 2)
    assert result["value"] == expected


def test_ca11e25_edge():
    """Test edge case with a different single positive se_d value."""
    se_d = 1.2
    result = ca_chapter_11_equation_25(se_d)
    assert isinstance(result, dict)
    assert "value" in result
    # Formula: se_lnOR = sqrt(se_d^2 / 0.551^2)
    expected = np.sqrt(se_d ** 2 / 0.551 ** 2)
    assert result["value"] == expected
