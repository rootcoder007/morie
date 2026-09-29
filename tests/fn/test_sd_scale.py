"""Tests for morie.fn.sd_scale: values recomputed from first principles."""

import math

from morie.fn.sd_scale import sd_scale


def test_scaling():
    # values of X scaled by a: population sd scales by |a|
    x = [1.0, 4.0, 2.0, 7.0]
    m = sum(x) / 4
    s = math.sqrt(sum((v - m) ** 2 for v in x) / 4)
    ax = [-2.5 * v for v in x]
    ma = sum(ax) / 4
    sa = math.sqrt(sum((v - ma) ** 2 for v in ax) / 4)
    assert abs(sd_scale(-2.5, s)["sd_aX"] - sa) < 1e-14
