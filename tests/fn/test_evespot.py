"""Tests for evespot.evt_pot_es."""

from morie.fn import _array_core as np

from morie.fn.evespot import evt_pot_es


def test_evespot_basic():
    """Test basic functionality."""
    u = 1.0
    sigma = 1.0
    xi = 0.5
    VaR = 3.0
    result = evt_pot_es(u, sigma, xi, VaR)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    # Independent computation from the documented formula
    expected_es = (VaR + sigma - xi * u) / (1.0 - xi)
    assert result["ES"] == expected_es
    assert result["estimate"] == expected_es
    assert result["xi"] == xi
    assert result["ratio"] == expected_es / VaR


def test_evespot_edge():
    """Test edge cases."""
    u = 2.0
    sigma = 0.5
    xi = 0.0  # xi = 0 collapses to VaR + sigma (memoryless exponential)
    VaR = 5.0
    result = evt_pot_es(u, sigma, xi, VaR)
    assert isinstance(result, dict)
    expected_es = VaR + sigma
    assert result["ES"] == expected_es
    assert result["estimate"] == expected_es
