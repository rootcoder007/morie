"""Tests for fzhok.fauzi_higher_order_kernel: the KDE and the kernel moments recomputed."""

import math

import pytest

from morie.fn.fzhok import fauzi_higher_order_kernel

X = [math.sin(1.3 * k) + 0.2 * k for k in range(30)]


def _phi(u):
    return math.exp(-u * u / 2) / math.sqrt(2 * math.pi)


def test_order4_estimate_and_moments():
    t, h = 1.1, 0.7
    ref = sum(0.5 * (3 - ((t - v) / h) ** 2) * _phi((t - v) / h) for v in X) / (30 * h)
    r = fauzi_higher_order_kernel(X, t=t, h=h)
    assert abs(r["estimate"] - ref) < 1e-14
    assert r["mu_r"] == -3.0
    assert abs(r["R_K"] - 27 / (32 * math.sqrt(math.pi))) < 1e-15


def test_order6_by_quadrature():
    r = fauzi_higher_order_kernel(X, t=0.5, h=0.9, order=6)

    def k6(u):
        return (15 - 10 * u * u + u**4) / 8 * _phi(u)

    step = 0.001
    grid = [-15 + step * i for i in range(30001)]
    mom = [math.fsum(u**j * k6(u) for u in grid) * step for j in (0, 2, 4, 6)]
    assert abs(mom[0] - 1) < 1e-12 and abs(mom[1]) < 1e-12 and abs(mom[2]) < 1e-11
    assert abs(r["mu_r"] - mom[3]) < 1e-9
    assert abs(r["R_K"] - math.fsum(k6(u) ** 2 for u in grid) * step) < 1e-12
    assert abs(r["estimate"] - sum(k6((0.5 - v) / 0.9) for v in X) / (30 * 0.9)) < 1e-14


def test_order2_is_gaussian_and_odd_rejected():
    r = fauzi_higher_order_kernel(X, t=0.0, h=0.5, order=2)
    assert abs(r["estimate"] - sum(_phi(-v / 0.5) for v in X) / 15) < 1e-14
    with pytest.raises(ValueError):
        fauzi_higher_order_kernel(X, order=3)
