"""Tests for rgsepix.rangayyan_separability_index."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_separability_index


def test_rgsepix_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = np.array([0] * 20 + [1] * 20)
    result = rangayyan_separability_index(X, y)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rgsepix_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = np.array([0] * 14 + [1] * 13 + [2] * 13)
    result = rangayyan_separability_index(X, y)
    assert isinstance(result, dict)
    assert len(result) > 0
