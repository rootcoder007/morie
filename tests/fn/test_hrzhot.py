"""Tests for hrzhot.horowitz_T_F_estimators."""

from morie.fn import _array_core as np

from morie.fn.hrzhot import horowitz_T_F_estimators


def test_hrzhot_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    x = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    bandwidth = 0.3
    beta_hat = rng.normal(0, 1, p)
    result = horowitz_T_F_estimators(x, y, bandwidth, beta_hat)
    assert isinstance(result, dict)
    assert "y_grid" in result
    assert "T_hat" in result
    assert "u_grid" in result
    assert "F_hat" in result
    assert "beta" in result
    assert "method" in result
    assert result["d"] == p
    assert result["n"] == n


def test_hrzhot_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    x = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    bandwidth = 0.3
    beta_hat = rng.normal(0, 1, p)
    y_grid = np.linspace(-2.0, 2.0, 10)
    u_grid = np.linspace(-2.0, 2.0, 10)
    y1 = 1.0
    y2 = -1.0
    result = horowitz_T_F_estimators(x, y, bandwidth, beta_hat,
                                     y_grid=y_grid, u_grid=u_grid,
                                     y1=y1, y2=y2)
    assert isinstance(result, dict)
    assert "y_grid" in result
    assert "T_hat" in result
    assert "u_grid" in result
    assert "F_hat" in result
    assert "window" in result
