"""Tests for kchnls.k_l_divergence_chain."""

from morie.fn import _array_core as np

from morie.fn.kchnls import k_l_divergence_chain


def test_kchnls_basic():
    """Test basic functionality."""
    pxy = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    qxy = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = k_l_divergence_chain(pxy, qxy)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kchnls_edge():
    """Test edge cases."""
    pxy = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    qxy = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = k_l_divergence_chain(pxy, qxy)
    assert isinstance(result, dict)
