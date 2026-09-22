"""Tests for evpot.evt_pot_fit."""

from morie.fn import _array_core as np

from morie.fn.evpot import evt_pot_fit


def test_evpot_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    u = 0.5
    result = evt_pot_fit(x, u)
    assert isinstance(result, dict)
    for key in ("sigma", "xi", "zeta_u", "estimate",
                "n_exceed", "n", "nll", "modified_scale"):
        assert key in result
    # n must equal the original sample size.
    assert result["n"] == len(x)
    # n_exceed is the count of observations strictly above u.
    y = sorted(v - u for v in x if v > u)
    assert result["n_exceed"] == len(y)
    # zeta_u = n_exceed / n.
    assert abs(result["zeta_u"] - len(y) / float(len(x))) < 1e-12
    # Independent computation of modified_scale from returned sigma, xi and u.
    assert abs(result["modified_scale"]
               - (result["sigma"] - result["xi"] * u)) < 1e-12


def test_evpot_edge():
    """Test that an explicit numeric threshold works (u must be scalar)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    u = 0.3
    result = evt_pot_fit(x, u)
    assert isinstance(result, dict)
    for key in ("sigma", "xi", "zeta_u", "estimate",
                "n_exceed", "n", "nll", "modified_scale"):
        assert key in result
    assert result["n"] == len(x)
    assert abs(result["modified_scale"]
               - (result["sigma"] - result["xi"] * u)) < 1e-12
