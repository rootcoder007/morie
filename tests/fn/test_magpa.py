"""Tests for magpa.ma_glmm_ipd_proportion."""

from morie.fn.magpa import ma_glmm_ipd_proportion


def test_magpa_basic():
    """Test basic functionality."""
    xi = 0.5
    ni = 0.5
    result = ma_glmm_ipd_proportion(xi, ni)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_magpa_edge():
    """Test edge cases."""
    xi = 0.5
    ni = 0.5
    result = ma_glmm_ipd_proportion(xi, ni)
    assert isinstance(result, dict)


def test_binomial_normal_likelihood_is_maximised():
    """Recompute the Gauss-Hermite marginal log-likelihood and check the
    reported (logit_mu, sigma) beats its neighbours."""
    import math

    import pytest

    from morie.fn.k02util import k02gh

    x = [3.0, 7.0, 12.0]
    n = [40.0, 50.0, 60.0]
    nodes, wts = k02gh(9)

    def ll(mu, s):
        tot = 0.0
        for xi, ni in zip(x, n):
            acc = 0.0
            for t, w in zip(nodes, wts):
                eta = mu + s * math.sqrt(2) * t
                p = 1 / (1 + math.exp(-eta))
                acc += w * p**xi * (1 - p) ** (ni - xi)
            tot += math.log(math.comb(int(ni), int(xi))) + math.log(acc / math.sqrt(math.pi))
        return tot

    r = ma_glmm_ipd_proportion(x, n, quad=9)
    mu, s = r["logit_mu"], r["sigma"]
    assert r["loglik"] == pytest.approx(ll(mu, s), rel=1e-9)
    for dm, ds in ((1e-3, 0), (-1e-3, 0), (0, 1e-3)):
        assert ll(mu + dm, s + ds) <= ll(mu, s) + 1e-9
    assert r["estimate"] == pytest.approx(1 / (1 + math.exp(-mu)), rel=1e-14)
