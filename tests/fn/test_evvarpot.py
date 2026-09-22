"""Tests for evvarpot.evt_pot_var."""

from morie.fn import _array_core as np

from morie.fn.evvarpot import evt_pot_var


def test_evvarpot_basic():
    """Test basic functionality against the documented formula."""
    u = 2.0
    sigma = 1.0
    xi = 0.5
    zeta_u = 0.1
    p = 0.99  # p must exceed 1 - zeta_u = 0.9

    r = (1.0 - p) / zeta_u
    expected_var = u + (sigma / xi) * (r ** (-xi) - 1.0)

    result = evt_pot_var(u, sigma, xi, zeta_u, p)

    assert isinstance(result, dict)
    assert "VaR" in result
    assert "estimate" in result
    assert "tail_prob" in result
    assert "p" in result
    assert result["p"] == p
    assert abs(result["VaR"] - expected_var) < 1e-12
    assert abs(result["estimate"] - expected_var) < 1e-12
    # Self-consistency: implied tail probability at the VaR should be ~ (1 - p)
    assert abs(result["tail_prob"] - (1.0 - p)) < 1e-9


def test_evvarpot_edge():
    """Test edge case xi = 0 (log branch)."""
    u = 1.0
    sigma = 2.0
    xi = 0.0
    zeta_u = 0.2
    p = 0.95  # > 1 - zeta_u = 0.8

    # xi -> 0 limit: u + sigma * log(zeta_u / (1 - p))
    expected_var = u + sigma * np.log(zeta_u / (1.0 - p))

    result = evt_pot_var(u, sigma, xi, zeta_u, p)

    assert isinstance(result, dict)
    assert abs(result["VaR"] - expected_var) < 1e-9
    assert abs(result["estimate"] - expected_var) < 1e-9
    assert abs(result["tail_prob"] - (1.0 - p)) < 1e-9
