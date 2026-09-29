"""Tests for morie.fn.xvrig: fan_out x fan_in Glorot weights recomputed from Philox draws."""

import math

import pytest

from morie.fn._rng import random_normal, random_uniform
from morie.fn.xvrig import xavier_init


def _l(v):
    return v.tolist() if hasattr(v, "tolist") else list(v)


def test_layout_and_normal_scale():
    z = _l(random_normal(6, seed=5))
    W = xavier_init(3, 2, seed=5).extra["weights"]
    assert len(W) == 2 and len(W[0]) == 3
    assert abs(W[1][2] - math.sqrt(2.0 / 5.0) * z[5]) < 1e-15


def test_uniform_and_default_seed():
    u = _l(random_uniform(6, seed=0))
    a = math.sqrt(6.0 / 5.0)
    W = xavier_init(3, 2, distribution="uniform").extra["weights"]
    assert abs(W[0][1] - (-a + 2 * a * u[1])) < 1e-15
    with pytest.raises(ValueError):
        xavier_init(3, 2, distribution="cauchy")
