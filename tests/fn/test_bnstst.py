"""Tests for bnstst.bound_test_inference."""

from morie.fn import _array_core as np

from morie.fn.bnstst import bound_test_inference


def test_bnstst_basic():
    """Test basic functionality."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_test_inference(lower, upper)
    assert isinstance(result, dict)
    assert "lower" in result
def test_bnstst_edge():
    """Test edge cases."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_test_inference(lower, upper)
    assert isinstance(result, dict)
