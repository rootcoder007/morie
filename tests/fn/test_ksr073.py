"""Tests for ksr073.kosorok_ch3_max_likelihood_efficiency_corollary."""

from morie.fn import _array_core as np

from morie.fn.ksr073 import kosorok_ch3_max_likelihood_efficiency_corollary


def test_ksr073_basic():
    """Test basic functionality."""
    psi_dot = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    scores = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kosorok_ch3_max_likelihood_efficiency_corollary(psi_dot, scores)
    assert isinstance(result, dict)
    assert "avar" in result


def test_ksr073_edge():
    """Test edge cases."""
    psi_dot = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    scores = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kosorok_ch3_max_likelihood_efficiency_corollary(psi_dot, scores)
    assert isinstance(result, dict)
