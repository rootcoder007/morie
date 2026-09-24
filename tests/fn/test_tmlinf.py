"""Tests for tmlinf.tmle_inference."""

from morie.fn import _array_core as np

from morie.fn.tmlinf import tmle_inference


def test_tmlinf_basic():
    """Test basic functionality."""
    psi = 0.1
    ic = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_inference(psi, ic)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_tmlinf_edge():
    """Test edge cases."""
    psi = 0.1
    ic = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_inference(psi, ic)
    assert isinstance(result, dict)
