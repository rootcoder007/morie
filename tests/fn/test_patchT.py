"""Tests for patchT.patch_tst."""

from morie.fn import _array_core as np

from morie.fn.patchT import patch_tst


def test_patchT_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    patch_len = 5
    result = patch_tst(X, patch_len)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_patchT_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    patch_len = 5
    result = patch_tst(X, patch_len)
    assert isinstance(result, dict)
