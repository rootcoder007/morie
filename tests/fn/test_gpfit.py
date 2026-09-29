"""Tests for morie.fn.gpfit: the GPD likelihood equations checked at the returned estimate."""

import math

import pytest

from morie.fn.gpfit import generalized_pareto, gpfit

X = [0.2, 1.4, 0.7, 2.9, 0.3, 5.1, 1.1, 0.9, 3.8, 0.5, 2.2, 7.4, 1.8, 0.6, 4.3, 0.4, 2.6, 9.9, 1.3, 0.8]


def _ll(y, s, k):
    return -len(y) * math.log(s) - (1 + 1 / k) * sum(math.log(1 + k * v / s) for v in y)


def test_likelihood_is_stationary_at_the_estimate():
    r = generalized_pareto(X, threshold=0.5)
    y = [v - 0.5 for v in X if v > 0.5]
    s, k = r["scale"], r["shape"]
    assert r["n_exceedances"] == len(y)
    for ds, dk in ((1e-4, 0), (0, 1e-4)):
        g = (_ll(y, s + ds, k + dk) - _ll(y, s - ds, k - dk)) / 2e-4
        assert abs(g) < 1e-6
    assert abs(r["loglik"] - _ll(y, s, k)) < 1e-10
    # it is a maximum: nearby points are lower
    assert _ll(y, s * 1.01, k) < r["loglik"] and _ll(y, s, k + 0.01) < r["loglik"]


def test_default_threshold_and_dispatch():
    Y = [v * f for v in X for f in (1.0, 1.7, 0.6)]
    s = sorted(Y)
    h = (len(s) - 1) * 0.9
    lo = math.floor(h)
    u = s[lo] + (h - lo) * (s[lo + 1] - s[lo])
    assert abs(generalized_pareto(Y)["threshold"] - u) < 1e-15
    r = gpfit(x=X, threshold=0.5)
    assert r["shape"] == generalized_pareto(X, threshold=0.5)["shape"]
    assert gpfit(data=X, threshold=0.5)["scale"] == r["scale"]
    with pytest.raises(ValueError):
        gpfit(data=X, coords=[[0, 1]] * 20)
