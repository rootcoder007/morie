"""Tests for morie.fn.sd_of_mean: values recomputed from first principles."""

import math

from morie.fn.sd_of_mean import sd_of_mean


def test_sigma_over_root_n():
    assert abs(sd_of_mean(3.0, 7)["sd_mean"] - 3.0 / math.sqrt(7)) < 1e-15
