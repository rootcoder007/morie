"""Tests for morie.fn.ksamp: the Scholz-Stephens statistic recomputed from its definition."""

import math

from morie.fn.ksamp import k_sample_anderson_darling

A = [0.1, 1.2, 0.5, 2.2, 1.9, 0.7, 1.2]
B = [3.1, 2.4, 4.2, 3.3, 2.9, 1.2]
C = [1.5, 0.9, 2.6, 2.0, 1.1]


def _a2akn(samples):
    # midrank (ties) A2akN of Scholz and Stephens (1987, eq. 7), written from the definition
    allv = sorted(v for s in samples for v in s)
    n, k = len(allv), len(samples)
    tot = 0.0
    for s in samples:
        inner = 0.0
        for z in sorted(set(allv)):
            lj = allv.count(z)
            bj = sum(1 for v in allv if v < z) + lj / 2
            mij = sum(1 for v in s if v < z) + s.count(z) / 2
            den = bj * (n - bj) - n * lj / 4
            if den > 0:
                inner += lj / n * (n * mij - bj * len(s)) ** 2 / den
        tot += inner / len(s)
    return tot * (n - 1) / n - (k - 1)


def _sigma(ns):
    n, k = sum(ns), len(ns)
    H = sum(1 / v for v in ns)
    h = sum(1 / i for i in range(1, n))
    g = sum(1 / ((n - i) * j) for i in range(1, n - 1) for j in range(i + 1, n))
    a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
    b = (2 * g - 4) * k * k + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
    c = (6 * h + 2 * g - 2) * k * k + (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
    d = (2 * h + 6) * k * k - 4 * h * k
    return math.sqrt((a * n**3 + b * n**2 + c * n + d) / ((n - 1) * (n - 2) * (n - 3)))


def test_standardised_statistic_with_ties():
    r = k_sample_anderson_darling(A, B, C)
    t = _a2akn([A, B, C]) / _sigma([7, 6, 5])
    assert abs(r.statistic - t) < 1e-12
    assert r.extra["k"] == 3 and r.n == 18


def test_p_value_bounds_and_ordering():
    far = k_sample_anderson_darling(A, [v + 5 for v in A])
    near = k_sample_anderson_darling(A, [v + 0.01 for v in A])
    assert far.p_value == 0.001 and near.p_value == 0.25
