"""Tests for hrzfpt.horowitz_first_passage_time."""

import math

from morie.fn import _array_core as np

from morie.fn.hrzfpt import horowitz_first_passage_time


def _std_normal_density(grid):
    return [math.exp(-v * v / 2.0) / math.sqrt(2.0 * math.pi) for v in grid]


def test_hrzfpt_basic():
    """Test basic functionality with standard normal densities for U and eps."""
    rng = np.random.default_rng(42)
    theta = 5
    d = 2
    y1 = 0.0
    y_star = 1.0

    x = rng.normal(0, 1, (theta, d))
    beta = rng.normal(0, 1, d)

    n_grid = 51
    grid_u = np.linspace(-3.0, 3.0, n_grid)
    f_U = _std_normal_density(grid_u)

    grid_z = np.linspace(-3.0, 3.0, n_grid)
    f_eps = _std_normal_density(grid_z)

    result = horowitz_first_passage_time(
        theta, y1, y_star, x, beta, f_U, grid_u, f_eps, grid_z
    )

    assert isinstance(result, dict)
    p = result["probability"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
    assert result["theta"] == theta
    assert math.isfinite(result["f_W_at_initial"])
    assert result["periods_conditionally_independent"] is True
    assert result["periods_marginally_independent"] is False
    assert isinstance(result["method"], str)


def test_hrzfpt_edge():
    """Test edge case: minimum valid horizon theta=2."""
    rng = np.random.default_rng(7)
    theta = 2
    d = 1
    y1 = 0.0
    y_star = 0.5

    x = rng.normal(0, 1, (theta, d))
    beta = rng.normal(0, 1, d)

    n_grid = 51
    grid_u = np.linspace(-3.0, 3.0, n_grid)
    f_U = _std_normal_density(grid_u)

    grid_z = np.linspace(-3.0, 3.0, n_grid)
    f_eps = _std_normal_density(grid_z)

    result = horowitz_first_passage_time(
        theta, y1, y_star, x, beta, f_U, grid_u, f_eps, grid_z
    )

    assert isinstance(result, dict)
    p = result["probability"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
    assert result["theta"] == 2
