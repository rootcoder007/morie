"""Tests for wsmmcm.wasserman_mcmc_metropolis."""

from morie.fn import _array_core as np

from morie.fn.wsmmcm import wasserman_mcmc_metropolis


def test_wsmmcm_basic():
    """Test basic functionality."""
    target = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    proposal = 0.1
    x0 = 0.1
    n = 5
    result = wasserman_mcmc_metropolis(target, proposal, x0, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmmcm_edge():
    """Test edge cases."""
    target = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    proposal = 0.1
    x0 = 0.1
    n = 5
    result = wasserman_mcmc_metropolis(target, proposal, x0, n)
    assert isinstance(result, dict)
