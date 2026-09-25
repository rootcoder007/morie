"""Tests for wnoma.wnominate_alternating (spatial roll-call scaling)."""

import math

import pytest

from morie.fn.wnoma import wnominate_alternating


def _votes(n=8, m=12):
    """Planted 1-D spatial voting: legislator i votes yea on roll call j
    when x_i is on roll call j's yea side of its cut point, with one
    flipped vote per legislator, and roll call 0 made unanimous."""
    x = [-1.5 + 3 * i / (n - 1) for i in range(n)]
    cuts = [-1.2 + 2.4 * j / (m - 1) for j in range(m)]
    V = [[1 if (xi > c) == (j % 2 == 0) else 0 for j, c in enumerate(cuts)] for xi in x]
    for i in range(n):
        V[i][(3 * i + 1) % m] = 1 - V[i][(3 * i + 1) % m]
    for i in range(n):
        V[i][0] = 1
    return V


def _pts(r):
    return [float(v[0]) if hasattr(v, "__len__") else float(v) for v in r["ideal_points"]]


def test_wnoma_basic():
    """The fit is normalised (mean 0, root-mean-square radius 1), puts the
    polarity legislator on the positive side, drops the unanimous roll
    call, classifies better than the modal baseline, and its likelihood
    does not decrease with more alternating steps."""
    V = _votes()
    r = wnominate_alternating(V, 1, polarity=7, max_iter=15)
    x = _pts(r)
    assert sum(x) / len(x) == pytest.approx(0.0, abs=1e-9)
    assert math.sqrt(sum(v * v for v in x) / len(x)) == pytest.approx(1.0, abs=1e-9)
    assert x[7] > 0
    assert r["n_dropped_rollcalls"] == 1
    assert r["correct_classification"] > r["modal_baseline"]
    r2 = wnominate_alternating(V, 1, polarity=7, max_iter=25)
    assert r2["log_likelihood"] >= r["log_likelihood"] - 1e-9


def test_wnoma_edge():
    """The polarity legislator decides the sign: naming the other end
    flips the configuration."""
    V = _votes()
    a = _pts(wnominate_alternating(V, 1, polarity=7, max_iter=10))
    b = _pts(wnominate_alternating(V, 1, polarity=0, max_iter=10))
    assert a == pytest.approx([-v for v in b], abs=1e-9)
