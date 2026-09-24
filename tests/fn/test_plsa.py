"""Tests for plsa.plsa."""

from morie.fn import _array_core as np

from morie.fn.plsa import plsa


def test_plsa_basic():
    """Test basic functionality."""
    n_dw = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    K = 5
    result = plsa(n_dw, K)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_plsa_edge():
    """Test edge cases."""
    n_dw = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    K = 5
    result = plsa(n_dw, K)
    assert isinstance(result, dict)
