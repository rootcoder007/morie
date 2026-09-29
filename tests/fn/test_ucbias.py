"""Tests for morie.fn.ucbias: the bounding factor and its link to the E-value."""

import math

from morie.fn.ucbias import unmeasured_conf_bias


def test_bias_factor():
    r = unmeasured_conf_bias(2.5, 1.8, RR_obs=1.6)
    B = 2.5 * 1.8 / (2.5 + 1.8 - 1)
    assert abs(r["bias_factor"] - B) < 1e-15
    assert abs(r["rr_bound"] - 1.6 / B) < 1e-15
    # at RR_UD = RR_UY = E-value the factor equals the observed RR
    e = 1.6 + math.sqrt(1.6 * 0.6)
    assert abs(unmeasured_conf_bias(e, e)["bias_factor"] - 1.6) < 1e-12
    assert unmeasured_conf_bias(e, e, RR_obs=1.6)["explains_away"]
