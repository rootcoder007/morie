"""Tests for sa_opt.simulated_annealing."""

from morie.fn import _array_core as np

from morie.fn.sa_opt import simulated_annealing


def test_sa_opt_basic():
    """Test basic functionality."""
    fun = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    x0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = simulated_annealing(fun, x0)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sa_opt_edge():
    """Test edge cases."""
    fun = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    x0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = simulated_annealing(fun, x0)
    assert isinstance(result, dict)
