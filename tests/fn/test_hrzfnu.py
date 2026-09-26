"""Tests for hrzfnu.horowitz_deconv_estimator (Horowitz sec. 5.2.1, eq. 5.26)."""

import math

import pytest

from morie.fn.hrzfnu import horowitz_deconv_estimator

N, TT = 30, 3
BETA = [0.7]
X = [[[math.sin(0.9 * j + 1.3 * t)] for t in range(TT)] for j in range(N)]
Y = [[BETA[0] * X[j][t][0] + math.cos(2.1 * j) + 0.3 * math.sin(3.7 * j + 5.1 * t) for t in range(TT)]
     for j in range(N)]
Wr = [Y[j][t] - BETA[0] * X[j][t][0] for j in range(N) for t in range(TT)]            # (5.21)
ETA = [(Y[j][t] - Y[j][0]) - BETA[0] * (X[j][t][0] - X[j][0][0]) for j in range(N) for t in range(1, TT)]  # (5.22)


def _zeta(u):
    """Fourfold convolution of the U[-1/4, 1/4] density, over its value
    4/3 at 0: the cubic B-spline (2 - |2u|... ) written as the Irwin-Hall
    density of 2u + 2 times 3/2."""
    if abs(u) >= 1:
        return 0.0
    x = 2 * abs(u) + 2
    return 1.5 * sum((-1) ** k * math.comb(4, k) * max(x - k, 0.0) ** 3 for k in range(5)) / 6


def _cf(v, t):
    return sum(math.cos(t * a) for a in v) / len(v), sum(math.sin(t * a) for a in v) / len(v)


def _trap(fn, lim):
    ts = [-lim + 2 * lim * k / 2000 for k in range(2001)]
    vals = [fn(t) for t in ts]
    return 2 * lim / 2000 * (sum(vals) - 0.5 * (vals[0] + vals[-1])) / (2 * math.pi)


def _fU(u, nu):
    def g(t):
        a, b = _cf(Wr, t)
        c, d = _cf(ETA, t)
        return (a * math.cos(t * u) + b * math.sin(t * u)) * _zeta(nu * t) / math.sqrt(math.hypot(c, d))
    return _trap(g, 1 / nu)


def _feps(z, nu):
    def g(t):
        c, d = _cf(ETA, t)
        return math.sqrt(math.hypot(c, d)) * _zeta(nu * t) * math.cos(t * z)
    return _trap(g, 1 / nu)


def test_hrzfnu_basic():
    """f_nU(u) = (1/2pi) int e^{-i tau u} psi_nW psi_zeta(nu tau) / |psi_n eta|^{1/2},
    recomputed from the residuals (5.21)-(5.22); psi_zeta is a real
    characteristic function on [-1, 1]: its inverse transform is
    (3/4) sinc^4(x/4) >= 0 and psi_zeta(0) = 1."""
    r = horowitz_deconv_estimator(Y, X, BETA, nu_U=0.5, grid=[0.0, 0.8])
    assert [float(v) for v in r["f_U"]] == pytest.approx([_fU(0.0, 0.5), _fU(0.8, 0.5)], rel=1e-9)
    assert r["cutoff"] == pytest.approx(2.0, rel=1e-15)
    assert (r["n"], r["T"]) == (N, TT)
    # 2001-point trapezoid of a C2 cubic spline: O(step^2) ~ 1e-6 quadrature error
    for x in (0.5, 3.0, 9.0):
        ft = 2 * math.pi * _trap(lambda t: _zeta(t) * math.cos(t * x), 1.0)
        assert ft == pytest.approx(0.75 * (math.sin(x / 4) / (x / 4)) ** 4, abs=1e-6)


def test_hrzfnu_edge():
    """A non-positive bandwidth and a one-period panel raise."""
    with pytest.raises(ValueError):
        horowitz_deconv_estimator(Y, X, BETA, nu_U=-1.0)
    with pytest.raises(ValueError):
        horowitz_deconv_estimator([[v[0]] for v in Y], [[v[0]] for v in X], BETA)


def test_hrzfnu_flattop():
    """kernel="flattop" uses psi_zeta = 1{|u| <= 1}; its default cut-off
    is the first tau (on a 0.02 / sd(W) grid) where |psi_nW| falls to
    2 / sqrt(N_W), and f_nU is the same inversion with that weight."""
    import statistics
    r = horowitz_deconv_estimator(Y, X, BETA, grid=[0.0, 0.8], kernel="flattop")
    step = 0.02 / statistics.stdev(Wr)
    floor = 2 / math.sqrt(len(Wr))
    T = next(k * step for k in range(1, 1501) if math.hypot(*_cf(Wr, k * step)) < floor)
    assert r["nu_U"] == pytest.approx(1 / T, rel=1e-12)

    def fu(u, nu):
        def g(t):
            a, b = _cf(Wr, t)
            c, d = _cf(ETA, t)
            return (a * math.cos(t * u) + b * math.sin(t * u)) / math.sqrt(math.hypot(c, d))
        return _trap(g, 1 / nu)
    assert [float(v) for v in r["f_U"]] == pytest.approx([fu(0.0, 1 / T), fu(0.8, 1 / T)], rel=1e-9)
