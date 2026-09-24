"""Tests for reinfc.reinforce."""

from morie.fn import _array_core as np

from morie.fn.reinfc import reinforce


def test_reinfc_basic():
    """Test basic functionality."""
    reward_fn = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    result = reinforce(reward_fn)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_reinfc_edge():
    """Test edge cases."""
    reward_fn = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    result = reinforce(reward_fn)
    assert isinstance(result, dict)
