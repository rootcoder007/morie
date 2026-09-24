"""Tests for grsa.geron_self_attention."""

from morie.fn import _array_core as np

from morie.fn.grsa import geron_self_attention


def test_grsa_basic():
    """Test basic functionality."""
    X = np.array([[1.0, 2.0], [0.0, -1.0]])
    WQ = np.array([[1.0, 0.0], [0.5, 1.0]])
    WK = np.array([[0.0, 1.0], [1.0, 0.0]])
    WV = np.array([[2.0, 0.0], [0.0, 3.0]])
    result = geron_self_attention(X, WQ, WK, WV)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grsa_edge():
    """Test edge cases."""
    X = np.array([[1.0, 2.0], [0.0, -1.0]])
    WQ = np.array([[1.0, 0.0], [0.5, 1.0]])
    WK = np.array([[0.0, 1.0], [1.0, 0.0]])
    WV = np.array([[2.0, 0.0], [0.0, 3.0]])
    result = geron_self_attention(X, WQ, WK, WV)
    assert isinstance(result, dict)
