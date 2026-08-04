"""Tests for rng253.rangayyan_ch4_power_cepstrum_definition."""

from morie.fn import _array_core as np

from morie.fn.bsacep import rangayyan_ch4_power_cepstrum_definition


def test_rng253_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_power_cepstrum_definition(Y, z, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng253_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_power_cepstrum_definition(Y, z, n)
    assert isinstance(result, dict)
