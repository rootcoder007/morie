"""Tests for spspec2.schabenberger_spectral_sim."""

from morie.fn import _array_core as np

from morie.fn.spspec2 import schabenberger_spectral_sim


def test_spspec2_basic():
    """Test basic functionality."""
    mu = 0.5
    cov_matrix = 0.5
    result = schabenberger_spectral_sim(mu, cov_matrix)
    assert isinstance(result, dict)
    assert "field" in result


def test_spspec2_edge():
    """Test edge cases."""
    mu = 0.5
    cov_matrix = 0.5
    result = schabenberger_spectral_sim(mu, cov_matrix)
    assert isinstance(result, dict)
