"""Tests for wmtwgt.weights_matrix."""

from morie.fn import _array_core as np

from morie.fn.wmtwgt import weights_matrix


def test_wmtwgt_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = weights_matrix(coords)
    assert isinstance(result, dict)
    assert "weights" in result


def test_wmtwgt_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = weights_matrix(coords)
    assert isinstance(result, dict)
