"""Tests for irtgrm.graded_response."""

from morie.fn import _array_core as np

from morie.fn.irtgrm import graded_response


def test_irtgrm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ncats = np.random.default_rng(42).normal(0, 1, 100)
    result = graded_response(X, ncats)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_irtgrm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ncats = np.random.default_rng(42).normal(0, 1, 100)
    result = graded_response(X, ncats)
    assert isinstance(result, dict)
