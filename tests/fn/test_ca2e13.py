"""Tests for ca2e13.ca_chapter_2_equation_13."""

from morie.fn import _array_core as np

from morie.fn.ca2e13 import ca_chapter_2_equation_13


def test_ca2e13_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_13(y, yhat)
    assert isinstance(result, dict)
    assert "var_total" in result
def test_ca2e13_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_13(y, yhat)
    assert isinstance(result, dict)
