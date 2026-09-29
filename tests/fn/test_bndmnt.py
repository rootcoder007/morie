"""Tests for bndmnt.bound_monotone_test: the variance-weighted statistic recomputed by brute force."""

import math

from morie.fn.bndmnt import bound_monotone_test

Z = [i % 2 for i in range(60)]
D = [1 if (math.sin(2.7 * i) + 0.8 * Z[i] > 0.3) else 0 for i in range(60)]
Y = [round(math.cos(1.3 * i) + D[i], 1) for i in range(60)]


def _p(sel, z):
    return sum(1 for i in range(60) if sel(i) and Z[i] == z) / 60


def test_statistic_by_brute_force():
    n, Tn, xi = 60, 15.0, 0.07
    best = -math.inf
    g = sorted(set(Y))
    cands = [
        (lambda i, a=a, b=b, d=d: a <= Y[i] <= b and D[i] == d, -1.0 if d == 1 else 1.0)
        for d in (0, 1)
        for a in g
        for b in g
        if b >= a
    ]
    cands.append((lambda i: D[i] == 0, 1.0))
    for sel, s in cands:
        q1, q0 = _p(sel, 1), _p(sel, 0)
        phi = s * (q1 / 0.5 - q0 / 0.5)
        v = Tn / n * (q1 / 0.25 - q1**2 / 0.125 + q0 / 0.25 - q0**2 / 0.125)
        best = max(best, math.sqrt(Tn) * phi / max(xi, math.sqrt(max(v, 0))))
    r = bound_monotone_test(Y, D, Z, n_boot=19)
    assert abs(r["statistic"] - best) < 1e-12
    assert 0 <= r["p_value"] <= 1


def test_detects_reversed_instrument():
    z = [i % 2 for i in range(200)]
    d = [1 if (math.sin(2.7 * i) + 0.8 * z[i] > 0.3) else 0 for i in range(200)]
    y = [round(2 * (math.cos(1.3 * i) + d[i])) / 2 for i in range(200)]
    assert bound_monotone_test(y, d, z, n_boot=49)["p_value"] > 0.5
    assert bound_monotone_test(y, [1 - v for v in d], z, n_boot=49)["p_value"] < 0.05
