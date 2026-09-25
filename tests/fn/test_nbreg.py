"""Tests for negbin_regression."""

from morie.fn import _array_core as np

import pytest

from morie.fn.nbreg import negbin_regression


def _counts():
    """Overdispersed counts with extra zeros, built without an RNG."""
    import math
    n = 120
    x1 = [math.sin(1.3 * i) for i in range(n)]
    x2 = [((i * 7) % 11 - 5) / 5 for i in range(n)]
    y = [0 if (i * 5) % 6 == 0 else
         math.floor(math.exp(0.6 + 0.5 * a - 0.3 * b) * ((i * 17) % 7) ** 2 / 6.0)
         for i, (a, b) in enumerate(zip(x1, x2))]
    return y, [[a, b] for a, b in zip(x1, x2)]


class TestNegBin:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (80, 1))
        y = rng.negative_binomial(5, 0.5, 80)
        r = negbin_regression(y, X)
        assert r.name == "negbin"
        assert r.extra["alpha"] > 0

    def test_aic(self):
        rng = np.random.default_rng(1)
        y = rng.negative_binomial(3, 0.3, 60)
        X = rng.normal(0, 1, (60, 1))
        r = negbin_regression(y, X)
        assert np.isfinite(r.extra["aic"])

    def test_matches_statsmodels(self):
        """NB2 MLE equals statsmodels 0.15 NegativeBinomial(nb2), fitted
        to a score of 5e-15, on the same data."""
        y, X = _counts()
        r = negbin_regression(y, X)
        ref = [1.0460712902261202, 0.5074744617808576, -0.4254986101853973]
        for got, want in zip(r.extra["coefficients"].values(), ref):
            assert got == pytest.approx(want, rel=1e-9)
        assert r.extra["alpha"] == pytest.approx(2.3309608130933293, rel=1e-9)
        assert r.extra["log_likelihood"] == pytest.approx(-256.08912059854845, rel=1e-12)
