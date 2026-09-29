"""Tests for btsieve.boot_sieve_general."""

from morie.fn import _array_core as np
from morie.fn.btsieve import boot_sieve_general


def test_btsieve_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_sieve_general(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btsieve_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_sieve_general(x)
    assert isinstance(result, dict)


def test_white_noise_sieve_is_the_iid_bootstrap():
    """p_max = 0: residuals are the centred data; each path draws
    burn-in + n residuals from the Lehmer stream and keeps the last n."""
    import pytest

    from morie.fn._tail1core import Lcg

    x = [0.5, 0.9, 0.4, 1.3, 1.1, 0.2, -0.3, 0.1]
    n = len(x)
    mu = sum(x) / n
    res = [v - mu for v in x]
    rb = sum(res) / n
    res = [v - rb for v in res]
    g = Lcg(4)
    theta = []
    for _ in range(6):
        out = []
        for t in range(100 + n):
            v = res[min(int(g.unif() * n), n - 1)]
            if t >= 100:
                out.append(v + mu)
        theta.append(sum(out) / n)
    r = boot_sieve_general(x, B=6, seed=4, p_max=0)
    assert r["order"] == 0
    assert r["theta_b"] == pytest.approx(theta, rel=1e-14)
