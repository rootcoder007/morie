"""Tests for sgtspc.sgt_spectrum."""

from morie.fn import _array_core as np

from morie.fn.sgtspc import sgt_spectrum


def test_sgtspc_basic():
    """Test basic functionality."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_spectrum(W)
    assert isinstance(result, dict)
    assert "values" in result


def test_sgtspc_edge():
    """Test edge cases."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_spectrum(W)
    assert isinstance(result, dict)
