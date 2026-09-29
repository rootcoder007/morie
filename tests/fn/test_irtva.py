"""Tests for irtva.irt_variance_legislator."""

import math

from morie.fn.irtva import irt_variance_legislator

C = [[math.sin(k) + j, math.cos(1.7 * k) * j] for k in range(25) for j in (1,)]
C = [[math.sin(k), math.cos(1.7 * k) * 2, 0.1 * k] for k in range(25)]


def test_column_variances():
    r = irt_variance_legislator(C)
    for j in range(3):
        col = [row[j] for row in C]
        m = sum(col) / 25
        assert abs(r.value[j] - sum((v - m) ** 2 for v in col) / 24) < 1e-14
    assert r.extra["n_legislators"] == 3 and r.extra["n_samples"] == 25
