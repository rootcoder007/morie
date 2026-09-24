"""Verification tests for information_theory_mackay24e9.mupostsg.

The expected values are recomputed from MacKay (2003) eq. (24.9)-(24.11) p. 320 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay24e9 import mupostsg


def test_mupostsg_is_normal_at_the_sample_mean_with_variance_sigma2_over_n():
    xbar, n, sigma = 3.4, 16, 2.0
    res = mupostsg(xbar, n, sigma)
    assert res["mean"] == pytest.approx(xbar, abs=1e-12)
    assert res["var"] == pytest.approx(sigma ** 2 / n, rel=1e-12)
    assert res["se"] == pytest.approx(sigma / math.sqrt(n), rel=1e-12)


def test_mupostsg_standard_error_falls_as_the_root_of_the_sample_size():
    a = mupostsg(0.0, 25, 1.0)["se"]
    b = mupostsg(0.0, 100, 1.0)["se"]
    assert a / b == pytest.approx(2.0, rel=1e-12)
