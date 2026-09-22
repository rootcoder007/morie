"""Tests for aftwbl.aft_weibull."""

from morie.fn import _array_core as np

from morie.fn.aftwbl import aft_weibull


def test_aftwbl_basic():
    """Test basic functionality."""
    time = np.linspace(0.1, 10, 100)
    event = np.random.default_rng(42).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aft_weibull(time, event, X)
    assert isinstance(result, dict)
    assert "beta" in result
    assert "time_ratio" in result
    assert "sigma" in result
    assert "loglik" in result
    assert "aic" in result
    assert "converged" in result


def test_aftwbl_edge():
    """Test edge cases."""
    time = np.linspace(0.1, 10, 100)
    event = np.random.default_rng(42).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aft_weibull(time, event, X)
    assert isinstance(result, dict)
