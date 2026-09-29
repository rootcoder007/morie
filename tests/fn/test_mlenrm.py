"""Tests for mlenrm (maximum-likelihood normal fit)."""

import math

import pytest

from morie.fn.mlenrm import mlenrm


def test_mle_normal_fit_and_loglik_recomputed():
    x = [2.1, 3.4, 1.9, 5.6, 2.8]
    n = 5
    mu = sum(x) / n
    s = math.sqrt(sum((v - mu) ** 2 for v in x) / n)
    ll = sum(-0.5 * math.log(2 * math.pi * s * s) - (v - mu) ** 2 / (2 * s * s) for v in x)
    r = mlenrm(x)
    assert r["mu"] == pytest.approx(mu, rel=1e-14)
    assert r["sigma"] == pytest.approx(s, rel=1e-13)
    assert r["sigma_unbiased"] == pytest.approx(s * math.sqrt(n / (n - 1)), rel=1e-13)
    assert r["loglik"] == pytest.approx(ll, rel=1e-12)
