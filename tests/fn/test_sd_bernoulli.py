"""Tests for morie.fn.sd_bernoulli: values recomputed from first principles."""

import math

from morie.fn.sd_bernoulli import sd_bernoulli


def test_bernoulli_sd():
    p = 0.3
    mu = p
    var = p * (1 - mu) ** 2 + (1 - p) * mu**2
    assert abs(sd_bernoulli(p)["sd"] - math.sqrt(var)) < 1e-15
