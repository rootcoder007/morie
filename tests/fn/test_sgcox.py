"""Tests for morie.fn.sgcox: Cox process given its realised intensity."""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.sgcox import cox_process, sgcox

FIELD = [[2.0, 5.0, 0.5], [1.0, 8.0, 3.0]]
WIN = (0.0, 3.0, 1.0, 3.0)


def test_replays_philox_inversion():
    u = [float(v) for v in random_uniform(4096, seed=11, stream=1000)]
    pos, pts = 0, []
    for iy in range(2):
        for ix in range(3):
            m = FIELD[iy][ix]
            v, pos = u[pos], pos + 1
            k, p = 0, math.exp(-m)
            F = p
            while v > F and p > 0:
                k += 1
                p *= m / k
                F += p
            for _ in range(k):
                pts.append((ix + u[pos], 1.0 + iy + u[pos + 1]))
                pos += 2
    r = cox_process(FIELD, WIN, seed=11)
    assert r.n_points == len(pts)
    assert max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(r.points, pts)) < 1e-14


def test_mean_count_is_integrated_intensity():
    counts = [cox_process(FIELD, WIN, seed=s).n_points for s in range(1, 401)]
    mean = sum(counts) / len(counts)
    lam = sum(sum(r) for r in FIELD)
    # Poisson(19.5): sd of the mean over 400 runs is sqrt(19.5 / 400) = 0.22
    assert cox_process(FIELD, WIN).expected_count == lam
    assert abs(mean - lam) < 5 * math.sqrt(lam / 400)


def test_alias_and_shape_check():
    assert sgcox is cox_process
    assert cox_process([[0.0, -1.0]], (0, 1, 0, 1)).n_points == 0
    with pytest.raises(ValueError):
        cox_process([[1.0, 2.0], [1.0]], WIN)
