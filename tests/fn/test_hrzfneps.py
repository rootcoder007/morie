"""Tests for hrzfneps.horowitz_panel_density_estimators (eqs. 5.25-5.26)."""

import math

import pytest

from morie.fn.hrzfneps import horowitz_panel_density_estimators

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


def test_hrzfneps_basic():
    """f_n eps (5.25) needs no division, f_nU (5.26) does; the two
    bandwidths are separate and both default to (log n)^{-1/2}."""
    r = horowitz_panel_density_estimators(Y, X, BETA, nu_U=0.5, nu_eps=0.4, grid_u=[0.3], grid_z=[0.0, 0.2])
    assert float(r["f_U"][0]) == pytest.approx(_fU(0.3, 0.5), rel=1e-9)
    assert [float(v) for v in r["f_eps"]] == pytest.approx([_feps(0.0, 0.4), _feps(0.2, 0.4)], rel=1e-9)
    assert r["f_eps_requires_division"] is False and r["f_U_requires_division"] is True
    d = horowitz_panel_density_estimators(Y, X, BETA, grid_u=[0.0], grid_z=[0.0])
    assert d["nu_U"] == d["nu_eps"] == pytest.approx(math.log(N) ** -0.5, rel=1e-15)


def test_hrzfneps_edge():
    """Fewer than 10 individuals raise."""
    with pytest.raises(ValueError):
        horowitz_panel_density_estimators(Y[:5], X[:5], BETA)
