"""Verification tests for information_theory_mackay24e6.gllsuff.

The expected values are recomputed from MacKay (2003) eq. (24.5)-(24.6) p. 319 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay24e6 import gllsuff


def test_gllsuff_matches_the_likelihood_computed_from_the_sample():
    data = [1.2, 0.8, 1.9, 2.4, 0.1]
    n = len(data)
    xbar = sum(data) / n
    s = sum((x - xbar) ** 2 for x in data)
    mu, sigma = 1.0, 0.9
    direct = sum(-math.log(math.sqrt(2.0 * math.pi) * sigma)
                 - (x - mu) ** 2 / (2.0 * sigma ** 2) for x in data)
    res = gllsuff(xbar, s, n, mu, sigma)
    assert res["loglik"] == pytest.approx(direct, abs=1e-12)


def test_gllsuff_is_maximised_at_the_sample_mean():
    xbar, s, n, sigma = 2.0, 4.0, 10, 1.0
    at_mean = gllsuff(xbar, s, n, xbar, sigma)["loglik"]
    for mu in (1.5, 1.9, 2.1, 2.5):
        assert gllsuff(xbar, s, n, mu, sigma)["loglik"] < at_mean
