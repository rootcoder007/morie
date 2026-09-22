"""Tests for eqhae.equating_haebara."""

from morie.fn import _array_core as np

from morie.fn.eqhae import equating_haebara


def test_eqhae_basic():
    """Test basic functionality."""
    a_ref = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_ref = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    a_focal = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_focal = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = equating_haebara(a_ref, b_ref, a_focal, b_focal)
    assert isinstance(result, dict)
    assert "A" in result
def test_eqhae_edge():
    """Test edge cases."""
    a_ref = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_ref = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    a_focal = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_focal = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = equating_haebara(a_ref, b_ref, a_focal, b_focal)
    assert isinstance(result, dict)
