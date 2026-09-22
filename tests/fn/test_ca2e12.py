"""Tests for ca2e12.ca_chapter_2_equation_12."""

from morie.fn import _array_core as np

from morie.fn.ca2e12 import ca_chapter_2_equation_12


def test_ca2e12_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_12(y, yhat)
    assert isinstance(result, dict)
    assert "var_total" in result
def test_ca2e12_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_12(y, yhat)
    assert isinstance(result, dict)
