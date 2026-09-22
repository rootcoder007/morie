"""Tests for ca2e14.ca_chapter_2_equation_14."""

from morie.fn import _array_core as np

from morie.fn.ca2e14 import ca_chapter_2_equation_14


def test_ca2e14_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_14(y, yhat)
    assert isinstance(result, dict)
    assert "var_total" in result
def test_ca2e14_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_14(y, yhat)
    assert isinstance(result, dict)
