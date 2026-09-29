"""Tests for morie.fn.mizval."""

import math

from morie.fn.mizval import mizval


def test_z_and_pvalues():
    mi, e, v = 0.4, -0.05, 0.02
    z = (mi - e) / math.sqrt(v)
    up = 0.5 * math.erfc(z / math.sqrt(2))
    assert abs(mizval(mi, e, v).statistic - z) < 1e-14
    assert abs(mizval(mi, e, v).p_value - up) < 1e-15
    assert abs(mizval(mi, e, v, alternative="less").p_value - (1 - up)) < 1e-14
    assert abs(mizval(mi, e, v, alternative="two.sided").p_value - 2 * up) < 1e-15
