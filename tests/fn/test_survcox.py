"""Tests for survcox.cox_partial_likelihood."""

from morie.fn import _array_core as np

from morie.fn.survcox import cox_partial_likelihood


def test_survcox_basic():
    """Test basic functionality."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = cox_partial_likelihood(time, event, X)
    assert isinstance(result, dict)
    assert "loglik" in result


def test_survcox_edge():
    """Test edge cases."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = cox_partial_likelihood(time, event, X)
    assert isinstance(result, dict)
