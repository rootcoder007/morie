"""Tests for kmalbi.kamath_alibi_bias."""

from morie.fn import _array_core as np

from morie.fn.kmalbi import kamath_alibi_bias


def test_kmalbi_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    slopes = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_alibi_bias(Q, K, V, slopes)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmalbi_edge():
    """Test edge cases."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    slopes = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_alibi_bias(Q, K, V, slopes)
    assert isinstance(result, dict)
