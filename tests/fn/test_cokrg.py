"""Tests for morie.fn.cokrg: cokriging through the LMC cokriging solver."""

import math

import pytest

from morie.fn._qpcore import solve
from morie.fn._rng import random_uniform
from morie.fn.cokrg import cokriging
from morie.fn.lmckrige import lmc_cokriging

U = [float(v) for v in random_uniform(80, seed=4)]
C = [(U[i], U[i + 20]) for i in range(20)]
X = [3 * t for t in U[40:60]]
Y = [a + 0.3 * t for a, t in zip(X, U[60:80])]
KW = dict(sill_p=2.0, range_p=1.5, sill_s=1.5, range_s=1.0, cross_sill=0.4, cross_range=1.2, nugget=0.1)


def _cov(h, a, b):
    if (a, b) == (0, 0):
        return (KW["sill_p"] - KW["nugget"]) * math.exp(-h / KW["range_p"]) + (KW["nugget"] if h == 0 else 0.0)
    if (a, b) == (1, 1):
        return (KW["sill_s"] - KW["nugget"]) * math.exp(-h / KW["range_s"]) + (KW["nugget"] if h == 0 else 0.0)
    return KW["cross_sill"] * math.exp(-h / KW["cross_range"])


def test_simple_cokriging_system_recomputed():
    pts = C + C
    var = [0] * 20 + [1] * 20
    z = X + Y
    M = [[_cov(math.dist(pts[i], pts[j]), var[i], var[j]) for j in range(40)] for i in range(40)]
    for t in ((0.5, 0.5), (0.1, 0.9)):
        c0 = [_cov(math.dist(p, t), v, 0) for p, v in zip(pts, var)]
        w = solve(M, c0)
        est = math.fsum(a * b for a, b in zip(w, z))
        se = math.sqrt(KW["sill_p"] - math.fsum(a * b for a, b in zip(w, c0)))
        r = cokriging(X, Y, C, t, **KW)
        assert r.estimate == pytest.approx(est, abs=1e-10) and r.se == pytest.approx(se, abs=1e-10)


def test_ordinary_and_multiple_targets():
    T = [(0.5, 0.5), (0.2, 0.3)]
    r = cokriging(X, Y, C, T, means=None, **KW)
    assert len(r.estimate) == 2 and r.method.startswith("Ordinary")
    lmc = [
        {"model": "Nug", "B": [[0.1, 0.0], [0.0, 0.1]]},
        {"model": "Exp", "range": 1.5, "B": [[1.9, 0.0], [0.0, 0.0]]},
        {"model": "Exp", "range": 1.0, "B": [[0.0, 0.0], [0.0, 1.4]]},
        {"model": "Exp", "range": 1.2, "B": [[0.0, 0.4], [0.4, 0.0]]},
    ]
    ref = lmc_cokriging(X + Y, C + C, [0] * 20 + [1] * 20, T, lmc)
    assert max(abs(a - b) for a, b in zip(r.estimate, ref.prediction)) < 1e-12
    # ordinary cokriging weights: primary sum to one, secondary to zero
    assert math.fsum(ref.weights[0][:20]) == pytest.approx(1.0, abs=1e-12)


def test_errors():
    with pytest.raises(ValueError):
        cokriging(X, Y[:5], C, (0.5, 0.5))
    with pytest.raises(ValueError):
        cokriging(X, Y, C, (0.5, 0.5, 0.5))
