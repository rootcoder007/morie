"""Tests for ptmcmc.parallel_tempering."""

from morie.fn import _array_core as np

from morie.fn.ptmcmc import parallel_tempering


def test_ptmcmc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    log_p = lambda x: -0.5 * float(x) ** 2
    temperatures = np.linspace(0.1, 2.0, 10)
    x0 = float(rng.normal(0, 1, 1))
    n_iter = 50
    result = parallel_tempering(log_p, temperatures, x0, n_iter)
    assert isinstance(result, dict)


def test_ptmcmc_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    log_p = lambda x: -0.5 * float(x) ** 2
    temperatures = np.linspace(0.1, 1.0, 3)
    x0 = float(rng.normal(0, 1, 1))
    n_iter = 1
    result = parallel_tempering(log_p, temperatures, x0, n_iter)
    assert isinstance(result, dict)
