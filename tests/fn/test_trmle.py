"""Tests for morie.fn.trmle — Transformation model MLE."""

import pytest

from morie.fn import _array_core as np

from morie.fn.trmle import trmle


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = X @ np.array([1, -0.5]) + rng.normal(0, 0.5, n)
    result = trmle(y, X, n_basis=3)
    assert isinstance(result, dict)
    for key in ("beta", "se", "t_stat", "pval", "basis_coefs", "log_likelihood", "n_obs"):
        assert key in result


def test_basis_coefs_monotone():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    y = X[:, 0] + rng.normal(0, 0.2, n)
    result = trmle(y, X, n_basis=4)
    coefs = result["basis_coefs"]
    assert all(coefs[i] <= coefs[i + 1] + 1e-6 for i in range(len(coefs) - 1))


def test_se_finite():
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 1))
    y = 2 * X[:, 0] + rng.normal(0, 0.3, n)
    result = trmle(y, X, n_basis=3)
    assert all(np.isfinite(s) for s in result["se"])


def _mlt_data(n=60):
    import math
    from statistics import NormalDist
    k = range(n)
    x1 = [math.sin(1.3 * i) for i in k]
    x2 = [math.cos(0.7 * i) for i in k]
    y = [math.exp(0.5 + 0.8 * a - 0.4 * b + 0.5 * NormalDist().inv_cdf(((i * 37 + 11) % 97 + 0.5) / 97))
         for i, a, b in zip(k, x1, x2)]
    return y, [[a, b] for a, b in zip(x1, x2)]


def test_matches_the_most_likely_transformation_likelihood():
    """The reported log-likelihood is sum log phi(h(y) - x'b) + log h'(y)
    recomputed from the returned Bernstein coefficients; theta is
    non-decreasing; the constrained KKT conditions hold (zero gradient
    in every free direction, no ascent into the constraint).  mlt::mlt
    with Bernstein_basis(order = 4, ui = "increasing"), negative = TRUE
    and support = range(y) gives beta 1.670779, -0.779041, standard
    errors 0.2417126, 0.1948940 and log-likelihood -78.4293493 here."""
    import math
    y, X = _mlt_data()
    r = trmle(y, X, n_basis=5)
    th, be = r["basis_coefs"], r["beta"]
    lo, hi = min(y), max(y)
    M = 4

    def ll(th, be):
        tot = 0.0
        for yi, xi in zip(y, X):
            s = (yi - lo) / (hi - lo)
            h = sum(math.comb(M, k) * s ** k * (1 - s) ** (M - k) * th[k] for k in range(5))
            hp = sum(M / (hi - lo) * math.comb(M - 1, k) * s ** k * (1 - s) ** (M - 1 - k) * (th[k + 1] - th[k])
                     for k in range(M))
            z = h - sum(a * b for a, b in zip(xi, be))
            tot += -0.5 * z * z - 0.5 * math.log(2 * math.pi) + math.log(hp)
        return tot

    L = ll(th, be)
    assert r["log_likelihood"] == pytest.approx(L, abs=1e-10)
    assert all(th[k] <= th[k + 1] for k in range(4))
    # no feasible single-coordinate move of 1e-6 increases the likelihood
    # by more than second-order noise
    for j in range(2):
        for sgn in (1, -1):
            b2 = list(be)
            b2[j] += sgn * 1e-6
            assert ll(th, b2) <= L + 1e-11
    for k in range(5):
        for sgn in (1, -1):
            t2 = list(th)
            t2[k] += sgn * 1e-6
            if all(t2[i] <= t2[i + 1] for i in range(4)):
                assert ll(t2, be) <= L + 1e-11
    # 5e-7: the mlt values are quoted to 7 digits; the unrounded ones
    # agree with these to 3e-8
    assert r["se"] == pytest.approx([0.2417127, 0.1948940], abs=5e-7)
    assert r["log_likelihood"] >= -78.4293493


def test_an_intercept_column_is_refused():
    y, X = _mlt_data()
    with pytest.raises(ValueError):
        trmle(y, [[1.0] + row for row in X])
