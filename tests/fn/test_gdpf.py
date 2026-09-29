"""Tests for morie.fn.gdpf: trade-off function and (eps, delta) conversion recomputed."""

import math

import pytest

from morie.fn.gdpf import gaussian_dp


def _Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _qnorm(p):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_trade_off_and_delta():
    mu, eps = 1.3, 0.8
    r = gaussian_dp(mu=mu, alpha=[0.05, 0.3], epsilon=eps)
    for a, t in zip([0.05, 0.3], r["trade_off"]):
        assert abs(t - _Phi(_qnorm(1 - a) - mu)) < 1e-12
    d = _Phi(-eps / mu + mu / 2) - math.exp(eps) * _Phi(-eps / mu - mu / 2)
    assert abs(r["delta"] - d) < 1e-13


def test_gaussian_mechanism_and_perfect_privacy():
    assert gaussian_dp(mech=(2.0, 4.0))["mu"] == 0.5
    r = gaussian_dp(mu=0.0, alpha=[0.2])
    assert abs(r["trade_off"][0] - 0.8) < 1e-12 and r["delta"] == 0.0
    with pytest.raises(ValueError):
        gaussian_dp()
