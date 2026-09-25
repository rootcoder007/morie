"""Tests for zero_inflated_negbin."""

from morie.fn import _array_core as np

import pytest

from morie.fn.zinb import _loglik_grad, zero_inflated_negbin


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


class TestZINB:
    def test_basic(self):
        rng = np.random.default_rng(0)
        y = np.concatenate([np.zeros(30), rng.negative_binomial(3, 0.3, 70)])
        X = rng.normal(0, 1, (100, 1))
        r = zero_inflated_negbin(y, X)
        assert r.name == "zinb"
        assert r.extra["zero_prob"] > 0

    def test_ll_finite(self):
        rng = np.random.default_rng(1)
        y = rng.poisson(2, 80).astype(float)
        y[:10] = 0
        X = rng.normal(0, 1, (80, 1))
        r = zero_inflated_negbin(y, X)
        assert np.isfinite(r.extra["log_likelihood"])

    def test_matches_statsmodels(self):
        """ZINB2 MLE equals statsmodels 0.15 ZeroInflatedNegativeBinomialP
        (p=2, constant inflation), fitted to a score of 6e-14."""
        y, X = _counts()
        r = zero_inflated_negbin(y, X).extra
        assert r["converged"]
        assert r["inflation_logit"] == pytest.approx(-0.37109418911129627, rel=1e-9)
        ref = [1.563815632237566, 0.5107242671290959, -0.48318696177486226]
        for got, want in zip(r["coefficients"].values(), ref):
            assert got == pytest.approx(want, rel=1e-9)
        assert r["alpha"] == pytest.approx(0.3894440306746898, rel=1e-9)
        assert r["log_likelihood"] == pytest.approx(-247.36249612353964, rel=1e-12)

    def test_poisson_limit(self):
        """As alpha -> 0 the likelihood tends to the zero-inflated Poisson
        one, with a gap of order alpha (it used to lose every digit to
        lgamma cancellation)."""
        import math
        y, X = _counts()
        Xi = [[1.0] + x for x in X]
        g, b = -0.9, [0.6, 0.7, -0.5]
        pi = 1.0 / (1.0 + math.exp(-g))
        zip_ll = 0.0
        for yi, x in zip(y, Xi):
            mu = math.exp(sum(c * v for c, v in zip(b, x)))
            zip_ll += (math.log(pi + (1 - pi) * math.exp(-mu)) if yi == 0 else
                       math.log(1 - pi) - mu + yi * math.log(mu) - math.lgamma(yi + 1))
        gaps = [_loglik_grad([g] + b + [math.log(a)], [float(v) for v in y], Xi)[0] - zip_ll
                for a in (1e-6, 1e-9)]
        assert abs(gaps[0]) < 1e-3
        assert abs(gaps[1]) < abs(gaps[0]) * 1e-2
