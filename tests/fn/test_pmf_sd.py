"""Tests for morie.fn.pmf_sd: values recomputed from first principles."""

import math

from morie.fn.pmf_sd import pmf_sd


def test_sd_is_the_root_of_the_second_central_moment():
    v, p = [1.0, 2.0, 6.0], [0.2, 0.5, 0.3]
    mu = sum(a * b for a, b in zip(v, p))
    var = sum(b * (a - mu) ** 2 for a, b in zip(v, p))
    r = pmf_sd(v, p)
    assert abs(r["mean"] - mu) < 1e-15
    assert abs(r["sd"] - math.sqrt(var)) < 1e-15
