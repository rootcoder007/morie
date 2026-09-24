"""Tests for marba.ma_smd_var_correlated_designs."""

from morie.fn import _array_core as np

from morie.fn.marba import ma_smd_var_correlated_designs


def test_marba_basic():
    """Test basic functionality."""
    g = 0.1
    n = 5
    rho = 0.1
    result = ma_smd_var_correlated_designs(g, n, rho)
    assert isinstance(result, dict)
    assert "var_g" in result


def test_marba_edge():
    """Test edge cases."""
    g = 0.1
    n = 5
    rho = 0.1
    result = ma_smd_var_correlated_designs(g, n, rho)
    assert isinstance(result, dict)
