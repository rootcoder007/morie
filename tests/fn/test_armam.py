"""Tests for morie.fn.armam -- ARMA(p,q) model."""

import pytest

from morie.fn import _array_core as np
from morie.fn.armam import arma_fit


class TestARMAFit:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(200)
        res = arma_fit(y, p=1, q=1)
        assert res.name == "arma_fit"
        assert "phi" in res.extra
        assert "theta" in res.extra

    def test_short_raises(self):
        with pytest.raises(ValueError):
            arma_fit(np.ones(5), p=1, q=1)

    def test_cheatsheet(self):
        from morie.fn.armam import cheatsheet

        assert isinstance(cheatsheet(), str)


def test_armam_residuals_sigma2_and_loglik_recomputed():
    """The returned phi/theta reproduce the conditional residual recursion,
    sigma2 and the conditional Gaussian log-likelihood, and they are a local
    maximum of it."""
    import math

    import pytest

    y = [0.8, 1.1, 0.3, -0.2, 0.5, 1.4, 0.9, -0.1, 0.2, 0.7, 1.0, 0.4, -0.3, 0.6, 1.2]
    r = arma_fit(y, p=1, q=1)
    n = len(y)
    mu = sum(y) / n

    def eps_of(phi, th):
        yc = [v - mu for v in y]
        e = [0.0] * n
        for t in range(1, n):
            e[t] = yc[t] - phi * yc[t - 1] - th * e[t - 1]
        return e

    def ll(phi, th):
        e = eps_of(phi, th)
        ss = sum(v * v for v in e[1:])
        T = n - 1
        s2 = ss / T
        return -(0.5 * T * math.log(2 * math.pi * s2) + ss / (2 * s2))

    phi, th = r.extra["phi"][0], r.extra["theta"][0]
    e = eps_of(phi, th)
    assert r.extra["sigma2"] == pytest.approx(sum(v * v for v in e[1:]) / (n - 1), rel=1e-12)
    assert r.extra["loglik"] == pytest.approx(ll(phi, th), rel=1e-10)
    for dp, dq in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)):
        assert ll(phi + dp, th + dq) <= ll(phi, th) + 1e-9
