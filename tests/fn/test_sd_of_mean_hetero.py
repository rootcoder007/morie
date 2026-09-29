"""Tests for morie.fn.sd_of_mean_hetero: values recomputed from first principles."""

import math

from morie.fn.sd_of_mean_hetero import sd_of_mean_hetero


def test_variance_of_the_average():
    s = [1.0, 2.0, 0.5, 3.0]
    assert abs(sd_of_mean_hetero(s)["sd_avg"] - math.sqrt(sum(v * v for v in s) / 16)) < 1e-15
