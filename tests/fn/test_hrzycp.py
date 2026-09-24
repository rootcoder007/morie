"""Tests for hrzycp.horowitz_conditional_prediction."""

import math

from morie.fn import _array_core as np
from morie.fn.hrzycp import horowitz_conditional_prediction


def _phi(u):
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(float(u) / math.sqrt(2.0)))


def _make_grids(n=60):
    y_grid = np.linspace(-3.0, 3.0, n)
    # Identity transformation: T(y) = y, strictly increasing on the grid
    T_hat = np.array([float(v) for v in y_grid])
    u_grid = np.linspace(-3.0, 3.0, n)
    # Standard normal CDF, non-decreasing and in [0, 1]
    F_hat = np.array([_phi(float(u)) for u in u_grid])
    return y_grid, T_hat, u_grid, F_hat


def _expected_keys():
    return (
        "probability", "quantile", "gamma", "u_gamma",
        "index", "mean_root_n_estimable",
        "quantile_root_n_estimable", "n_points", "method",
    )


def _to_finite_list(val):
    """Convert a scalar or array to a list of floats, checking finiteness."""
    if hasattr(val, "__len__"):
        items = [float(v) for v in val]
    else:
        items = [float(val)]
    for v in items:
        assert math.isfinite(v)
    return items


def test_hrzycp_basic():
    """Test basic functionality with a single x and a single y threshold."""
    rng = np.random.default_rng(42)
    d = 3
    x = rng.normal(0, 1, d)
    beta_hat = rng.normal(0, 1, d)

    y_grid, T_hat, u_grid, F_hat = _make_grids()

    result = horowitz_conditional_prediction(
        x, 0.5, T_hat, F_hat, beta_hat,
        gamma=0.5,
        y_grid=y_grid, u_grid=u_grid,
    )

    assert isinstance(result, dict)
    for key in _expected_keys():
        assert key in result

    prob_items = _to_finite_list(result["probability"])
    assert all(0.0 <= p <= 1.0 for p in prob_items)

    assert result["gamma"] == 0.5
    _to_finite_list(result["u_gamma"])
    _to_finite_list(result["quantile"])
    _to_finite_list(result["index"])

    assert result["mean_root_n_estimable"] is False
    assert result["quantile_root_n_estimable"] is True

    assert isinstance(result["method"], str)


def test_hrzycp_edge():
    """Test with multiple prediction points (2-D x) and a low quantile level."""
    rng = np.random.default_rng(7)
    d = 3
    x = rng.normal(0, 1, (5, d))
    beta_hat = rng.normal(0, 1, d)

    y_grid, T_hat, u_grid, F_hat = _make_grids()

    result = horowitz_conditional_prediction(
        x, -1.0, T_hat, F_hat, beta_hat,
        gamma=0.1,
        y_grid=y_grid, u_grid=u_grid,
    )

    assert isinstance(result, dict)
    for key in _expected_keys():
        assert key in result

    assert result["gamma"] == 0.1

    # With (m, d) x, outputs have one entry per row.
    m = 5
    prob_items = _to_finite_list(result["probability"])
    assert len(prob_items) == m
    assert all(0.0 <= p <= 1.0 for p in prob_items)

    quant_items = _to_finite_list(result["quantile"])
    assert len(quant_items) == m

    idx_items = _to_finite_list(result["index"])
    assert len(idx_items) == m

    _to_finite_list(result["u_gamma"])

    assert result["mean_root_n_estimable"] is False
    assert result["quantile_root_n_estimable"] is True
