"""Tests for missinM.missing_mechanism_sensitivity."""

import math

from morie.fn import _array_core as np

from morie.fn.missinM import missing_mechanism_sensitivity


def test_missinM_basic():
    """Test basic functionality."""
    n = 100
    rng_y = np.random.default_rng(43)
    Y = rng_y.normal(0, 1, n)
    # Binary response indicator: 60 observed, 40 missing
    R = [1] * 60 + [0] * 40
    # A grid of shift parameters
    delta_grid = np.linspace(-2.0, 2.0, 11)

    # Compute expected quantities from the inputs
    obs_idx = [i for i in range(n) if R[i] == 1]
    mar_mean = sum(Y[i] for i in obs_idx) / len(obs_idx)
    p1 = len(obs_idx) / n
    p0 = 1.0 - p1
    reference = 0.0

    result = missing_mechanism_sensitivity(Y, R, delta_grid)

    # Check return type and required keys
    assert isinstance(result, dict)
    expected_keys = {
        "estimate", "means", "delta_grid", "mar_mean",
        "p_observed", "tipping_delta", "n_observed", "n"
    }
    assert expected_keys.issubset(result.keys())

    # Numerical checks
    assert math.isclose(result["estimate"], mar_mean, rel_tol=1e-12)
    assert math.isclose(result["mar_mean"], mar_mean, rel_tol=1e-12)
    assert math.isclose(result["p_observed"], p1, rel_tol=1e-12)
    assert result["n_observed"] == len(obs_idx)
    assert result["n"] == n
    assert len(result["delta_grid"]) == len(delta_grid)
    assert len(result["means"]) == len(delta_grid)

    # means follow the linear shift model
    for d, m in zip(delta_grid, result["means"]):
        expected_mean = mar_mean + p0 * d
        assert math.isclose(m, expected_mean, rel_tol=1e-12)

    # tipping point
    expected_tipping = (reference - mar_mean) / p0
    assert math.isclose(result["tipping_delta"], expected_tipping, rel_tol=1e-12)


def test_missinM_edge():
    """Test edge case with no missing data."""
    n = 50
    rng_y = np.random.default_rng(7)
    Y = rng_y.normal(0, 1, n)
    R = [1] * n  # all outcomes observed
    delta_grid = np.linspace(-1.0, 1.0, 5)

    obs_idx = [i for i in range(n) if R[i] == 1]
    mar_mean = sum(Y[i] for i in obs_idx) / len(obs_idx)

    result = missing_mechanism_sensitivity(Y, R, delta_grid)

    assert isinstance(result, dict)
    # With no missingness, p_observed = 1.0
    assert math.isclose(result["p_observed"], 1.0, rel_tol=1e-12)
    # Tipping point should be NaN (not infinity)
    assert math.isnan(result["tipping_delta"])
    # The marginal mean does not depend on delta when p0 = 0
    for m in result["means"]:
        assert math.isclose(m, mar_mean, rel_tol=1e-12)
    # The estimate equals the observed mean
    assert math.isclose(result["estimate"], mar_mean, rel_tol=1e-12)
