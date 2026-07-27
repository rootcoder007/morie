"""Tests for ucbias.unmeasured_conf_bias."""

import pytest

from morie.fn.ucbias import unmeasured_conf_bias


def test_ucbias_basic():
    # B = RR_UD * RR_UY / (RR_UD + RR_UY - 1) = 9 / 5 = 1.8
    out = unmeasured_conf_bias(3.0, 3.0, RR_obs=2.0)
    assert out["bias_factor"] == pytest.approx(1.8)
    assert out["rr_bound"] == pytest.approx(2.0 / 1.8)
    assert out["explains_away"] is False


def test_ucbias_edge():
    # the diagonal identity: at RR_UD = RR_UY = E-value, B equals the RR
    e = 2 + 2**0.5
    assert unmeasured_conf_bias(e, e)["bias_factor"] == pytest.approx(2.0)
    # a confounder that strong does explain away an observed RR of 2
    assert unmeasured_conf_bias(e, e, RR_obs=2.0)["explains_away"] is True
    with pytest.raises(ValueError):
        unmeasured_conf_bias(0.5, 3.0)  # RR_UD below 1
