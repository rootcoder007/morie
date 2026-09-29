"""Tests for morie.fn.mcerr -- MCMC standard error."""

from morie.fn import _array_core as np
from morie.fn.mcerr import mcmc_se


def test_returns_dict():
    samples = np.random.default_rng(42).standard_normal(500)
    result = mcmc_se(samples)
    assert isinstance(result, dict)
    assert "mcse" in result


def test_mcse_positive():
    samples = np.random.default_rng(42).standard_normal(500)
    result = mcmc_se(samples)
    assert result["mcse"] > 0


def test_mcse_smaller_than_sd():
    samples = np.random.default_rng(42).standard_normal(1000)
    result = mcmc_se(samples)
    assert result["mcse"] < result["sd"]


def test_batch_means_mcse_recomputed():
    import math

    import pytest

    x = [0.5, 0.9, 0.4, 1.3, 1.1, 0.2, -0.3, 0.1, 0.8, 0.6, 1.5, 1.0, 0.7, 0.3, 0.9, 1.2, 0.4]
    n = 17
    nb = int(math.sqrt(n))
    bs = n // nb
    bm = [sum(x[b * bs : (b + 1) * bs]) / bs for b in range(nb)]
    m = sum(bm) / nb
    r = mcmc_se(x)
    assert r["n_batches"] == 4
    assert r["mcse"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in bm) / (nb - 1)) / math.sqrt(nb), rel=1e-12)
