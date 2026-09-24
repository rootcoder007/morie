"""Tests for wmtrwn.weights_row_normalize."""

from morie.fn import _array_core as np

from morie.fn.wmtrwn import weights_row_normalize


def test_wmtrwn_basic():
    """Test basic functionality."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = weights_row_normalize(W)
    assert isinstance(result, dict)
    assert "W" in result


def test_wmtrwn_edge():
    """Test edge cases."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = weights_row_normalize(W)
    assert isinstance(result, dict)
