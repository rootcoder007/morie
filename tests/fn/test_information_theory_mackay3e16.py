"""Verification tests for information_theory_mackay3e16.sucrule.

The expected values are recomputed from MacKay (2003) eq. (3.16) p. 52 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay3e16 import sucrule


def test_sucrule_is_laplaces_rule_of_succession():
    for fa, fb in ((0, 0), (3, 0), (5, 7), (40, 2)):
        res = sucrule(fa, fb)
        assert res["p"] == pytest.approx((fa + 1.0) / (fa + fb + 2.0), abs=1e-12)
        assert res["pnot"] == pytest.approx(1.0 - res["p"], abs=1e-12)


def test_sucrule_with_no_data_gives_one_half_and_no_maximum_likelihood():
    res = sucrule(0, 0)
    assert res["p"] == pytest.approx(0.5, abs=1e-12)
    assert math.isnan(res["mle"])


def test_sucrule_approaches_the_sample_proportion_as_data_accumulate():
    far = sucrule(600, 400)
    assert abs(far["p"] - 0.6) < 1e-3
