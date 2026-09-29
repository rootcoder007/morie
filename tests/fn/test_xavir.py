"""Tests for morie.fn.xavir: Glorot uniform / normal weights recomputed from Philox draws."""

import math

from morie.fn._rng import random_normal, random_uniform
from morie.fn.xavir import xavier_init


def _l(v):
    return v.tolist() if hasattr(v, "tolist") else list(v)


def test_uniform_limits_and_draws():
    a = math.sqrt(6.0 / 7.0)
    u = _l(random_uniform(12, seed=2))
    r = xavier_init(4, 3, seed=2)
    W = r.extra["weights"]
    assert max(abs(W[i][j] - (-a + 2 * a * u[3 * i + j])) for i in range(4) for j in range(3)) < 1e-15
    flat = [v for row in W for v in row]
    m = sum(flat) / 12
    assert abs(r.value - math.sqrt(sum((v - m) ** 2 for v in flat) / 12)) < 1e-15


def test_normal_draws():
    z = _l(random_normal(6, seed=4))
    W = xavier_init(2, 3, seed=4, uniform=False).extra["weights"]
    assert abs(W[1][2] - math.sqrt(2.0 / 5.0) * z[5]) < 1e-15
