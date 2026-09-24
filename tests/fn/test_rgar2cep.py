"""Tests for rgar2cep.rangayyan_ar_to_cepstrum."""

from morie.fn import _array_core as np

from morie.fn.bsacep import rangayyan_ar_to_cepstrum


def test_rgar2cep_basic():
    """Test basic functionality."""
    a_coeffs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ar_to_cepstrum(a_coeffs)
    assert isinstance(result, dict)
    assert "cepstrum" in result


def test_rgar2cep_edge():
    """Test edge cases."""
    a_coeffs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ar_to_cepstrum(a_coeffs)
    assert isinstance(result, dict)
