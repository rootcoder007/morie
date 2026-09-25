"""Tests for tmlcou.tmle_count_outcome (TMLE of a bounded / count outcome)."""

import math
import statistics

import pytest

from morie.fn.tmlcou import (linear_fluctuation_unsafe, rescale,
                             tmle_count_outcome, unscale)


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p):
    return math.log(p / (1.0 - p))


def _data(n=50):
    W = [math.sin(1.9 * k) for k in range(n)]
    A = [1.0 if ((31 * k + 3) % 71 + 0.5) / 71.0 < _expit(0.4 * w) else 0.0 for k, w in enumerate(W)]
    Y = [float((13 * k + 5) % 7 + int(2 * a) + (1 if w > 0 else 0)) for k, (a, w) in enumerate(zip(A, W))]
    g = [_expit(0.1 + 0.5 * w) for w in W]
    Q1 = [4.5 + 0.8 * w for w in W]
    Q0 = [3.0 + 0.8 * w for w in W]
    return Y, A, [[w] for w in W], g, Q1, Q0


def _tmle(Y, A, g, Q1, Q0):
    """Gruber & van der Laan (2010): scale to [0,1] with the sample range,
    fluctuate logit Q(A,W) on H = A/g - (1-A)/(1-g); the score in epsilon
    is monotone, so bisection finds its root."""
    lo, hi = min(Y), max(Y)
    rg = hi - lo
    ys = [(y - lo) / rg for y in Y]
    q1 = [(q - lo) / rg for q in Q1]
    q0 = [(q - lo) / rg for q in Q0]
    H = [a / p - (1 - a) / (1 - p) for a, p in zip(A, g)]
    qa = [x if a else z for a, x, z in zip(A, q1, q0)]

    def score(e):
        return sum(h * (y - _expit(_logit(q) + e * h)) for h, y, q in zip(H, ys, qa))
    a_, b_ = -50.0, 50.0
    for _ in range(200):
        m = 0.5 * (a_ + b_)
        a_, b_ = (m, b_) if score(m) > 0 else (a_, m)
    e = 0.5 * (a_ + b_)
    q1s = [_expit(_logit(q) + e / p) for q, p in zip(q1, g)]
    q0s = [_expit(_logit(q) - e / (1 - p)) for q, p in zip(q0, g)]
    ps = statistics.fmean(x - z for x, z in zip(q1s, q0s))
    d = [(h * (y - (x if a else z)) + x - z - ps) * rg
         for h, y, a, x, z in zip(H, ys, A, q1s, q0s)]
    return ps * rg, e, statistics.pstdev(d) / math.sqrt(len(d))


def test_tmlcou_basic():
    """With supplied nuisances the estimate, epsilon and IC standard
    error equal an independent bisection TMLE on the rescaled outcome."""
    Y, A, X, g, Q1, Q0 = _data()
    psi, e, se = _tmle(Y, A, g, Q1, Q0)
    r = tmle_count_outcome(Y, A, X, g=g, Q1=Q1, Q0=Q0)
    assert r["estimate"] == pytest.approx(psi, abs=1e-9)
    assert r["epsilon"] == pytest.approx(e, abs=1e-9)
    assert r["se"] == pytest.approx(se, rel=1e-9)
    assert r["in_range"] and r["solves_eic"]


def test_tmlcou_edge():
    """Negative counts raise; the offset turns counts into rates; the
    affine map round-trips; the linear fluctuation can leave [0,1]."""
    Y, A, X, g, Q1, Q0 = _data()
    with pytest.raises(ValueError):
        tmle_count_outcome([-1.0] + Y[1:], A, X)
    t = [1.0 + (k % 3) for k in range(len(Y))]
    rate = [y / s for y, s in zip(Y, t)]
    Q1r, Q0r = [q / 2 for q in Q1], [q / 2 for q in Q0]
    r = tmle_count_outcome(Y, A, X, offset=t, g=g, Q1=Q1r, Q0=Q0r)
    assert r["estimate"] == pytest.approx(_tmle(rate, A, g, Q1r, Q0r)[0], abs=1e-9)
    assert r["rate_scale"] is True
    s = rescale([2.0, 4.0, 6.0])
    assert s["scaled"] == [0.0, 0.5, 1.0]
    assert unscale(0.5, s["lower"], s["upper"]) == 4.0
    lf = linear_fluctuation_unsafe([0.9, 0.95], [5.0, 5.0], [1.0, 1.0])
    # e = sum h(y-q)/sum h^2 = 5*0.15/50 = 0.015; update 0.9+0.075, 0.95+0.075
    assert lf["epsilon"] == pytest.approx(0.015, abs=1e-12)
    assert lf["out_of_range"] == 1
    d = tmle_count_outcome(Y, A, X)
    assert d["solves_eic"] and d["in_range"]
