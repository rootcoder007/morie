"""Tests for gpsvi.gp_stochastic_vi."""

from morie.fn import _array_core as np

from morie.fn.gpsvi import gp_stochastic_vi


def test_gpsvi_basic():
    """Test basic functionality."""
    rng = np.random.default_rng
    X = [[float(v) for v in row] for row in rng(42).normal(0, 1, (100, 5))]
    y = [float(v) for v in rng(43).normal(0, 1, 100)]
    X_test = [[float(v) for v in row] for row in rng(44).normal(0, 1, (30, 5))]
    inducing = [[float(v) for v in row] for row in rng(45).normal(0, 1, (10, 5))]
    result = gp_stochastic_vi(X, y, X_test, inducing)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mean" in result
    assert "variance" in result
    assert "elbo" in result
    assert "n" in result
    assert result["n"] == 100
    assert len(result["mean"]) == 30
    assert len(result["variance"]) == 30
    assert result["method"] == "collapsed sparse bound of Titsias (2009) as used by Hensman, Fusi & Lawrence (2013)"


def test_gpsvi_edge():
    """Test edge cases."""
    rng = np.random.default_rng
    X = [[float(v) for v in row] for row in rng(42).normal(0, 1, (100, 5))]
    y = [float(v) for v in rng(43).normal(0, 1, 100)]
    X_test = [[float(v) for v in row] for row in rng(44).normal(0, 1, (30, 5))]
    inducing = [[float(v) for v in row] for row in rng(45).normal(0, 1, (10, 5))]
    result = gp_stochastic_vi(X, y, X_test, inducing)
    assert isinstance(result, dict)
    assert "mean" in result
    assert "variance" in result
    assert "elbo" in result
    assert result["n"] == 100
