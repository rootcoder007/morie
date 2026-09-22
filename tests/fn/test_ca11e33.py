"""Tests for ca11e33.ca_chapter_11_equation_33."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.ca11e33 import ca_chapter_11_equation_33


def test_ca11e33_basic():
    """Test basic functionality."""
    d = 0.5
    se_d = 0.1
    result = ca_chapter_11_equation_33(d, se_d)
    assert isinstance(result, dict)
    assert "value" in result

    expected = (4 * se_d ** 2 / (d ** 2 + 4) ** 3) ** 0.5
    assert np.isclose(result["value"], expected)


def test_ca11e33_edge():
    """Test edge cases."""
    d = 1.0
    se_d = 0.2
    result = ca_chapter_11_equation_33(d, se_d)
    assert isinstance(result, dict)
    assert "value" in result

    expected = (4 * se_d ** 2 / (d ** 2 + 4) ** 3) ** 0.5
    assert np.isclose(result["value"], expected)

    d2 = 0.0
    se_d2 = 0.5
    result2 = ca_chapter_11_equation_33(d2, se_d2)
    expected2 = (4 * se_d2 ** 2 / (d2 ** 2 + 4) ** 3) ** 0.5
    assert np.isclose(result2["value"], expected2)

    d3 = 2.5
    se_d3 = 0.3
    result3 = ca_chapter_11_equation_33(d3, se_d3)
    expected3 = (4 * se_d3 ** 2 / (d3 ** 2 + 4) ** 3) ** 0.5
    assert np.isclose(result3["value"], expected3)

    assert "method" in result
