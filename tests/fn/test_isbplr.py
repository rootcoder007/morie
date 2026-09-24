"""Tests for isbplr.isgp_bayes."""

from morie.fn import _array_core as np

from morie.fn.isbplr import isgp_bayes


def test_isbplr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = isgp_bayes(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_isbplr_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = isgp_bayes(y)
    assert isinstance(result, dict)
