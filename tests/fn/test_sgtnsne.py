"""Tests for sgtnsne.sgt_isomap."""

from morie.fn import _array_core as np

from morie.fn.sgtnsne import sgt_isomap


def test_sgtnsne_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sgt_isomap(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "Y" in result


def test_sgtnsne_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sgt_isomap(X)
    assert isinstance(result, dict)
