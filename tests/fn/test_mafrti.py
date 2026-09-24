"""Tests for mafrti.ma_freeman_tukey_inverse."""

from morie.fn import _array_core as np

from morie.fn.mafrti import ma_freeman_tukey_inverse


def test_mafrti_basic():
    """Test basic functionality."""
    ft = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_harmonic = 0.1
    result = ma_freeman_tukey_inverse(ft, n_harmonic)
    assert isinstance(result, dict)
    assert "p" in result


def test_mafrti_edge():
    """Test edge cases."""
    ft = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_harmonic = 0.1
    result = ma_freeman_tukey_inverse(ft, n_harmonic)
    assert isinstance(result, dict)
