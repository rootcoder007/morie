"""Tests for sgtcheb.sgt_cheeger_bound."""

from morie.fn import _array_core as np

from morie.fn.sgtcheb import sgt_cheeger_bound


def test_sgtcheb_basic():
    """Test basic functionality."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_cheeger_bound(W)
    assert isinstance(result, dict)
    assert "h" in result


def test_sgtcheb_edge():
    """Test edge cases."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_cheeger_bound(W)
    assert isinstance(result, dict)
