"""Tests for gphtr.gp_heteroscedastic."""

from morie.fn import _array_core as np

from morie.fn.gphtr import gp_heteroscedastic


def test_gphtr_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_xt = np.random.default_rng(44)
    X = [[float(x) for x in row] for row in rng_x.normal(0, 1, (5, 5))]
    y = [float(v) for v in rng_y.normal(0, 1, 5)]
    X_test = [[float(x) for x in row] for row in rng_xt.normal(0, 1, (3, 5))]
    result = gp_heteroscedastic(X, y, X_test)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mean" in result
    assert "variance" in result
    assert "noise" in result
    assert "noise_test" in result
    assert "loglik" in result
    assert isinstance(result["mean"], list)
    assert isinstance(result["variance"], list)
    assert isinstance(result["noise"], list)
    assert isinstance(result["noise_test"], list)
    assert len(result["mean"]) == len(X_test)
    assert len(result["variance"]) == len(X_test)
    assert len(result["noise_test"]) == len(X_test)


def test_gphtr_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_xt = np.random.default_rng(44)
    X = [[float(x) for x in row] for row in rng_x.normal(0, 1, (5, 5))]
    y = [float(v) for v in rng_y.normal(0, 1, 5)]
    X_test = [[float(x) for x in row] for row in rng_xt.normal(0, 1, (3, 5))]
    result = gp_heteroscedastic(X, y, X_test)
    assert isinstance(result, dict)
