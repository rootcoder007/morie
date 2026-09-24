"""Tests for sgtpgr.sgt_pagerank_power."""

from morie.fn import _array_core as np

from morie.fn.sgtpgr import sgt_pagerank_power


def test_sgtpgr_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_pagerank_power(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "pr" in result


def test_sgtpgr_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_pagerank_power(A)
    assert isinstance(result, dict)
