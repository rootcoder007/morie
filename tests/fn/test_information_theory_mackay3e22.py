"""Verification tests for information_theory_mackay3e22.bcoinbf.

The expected values are recomputed from MacKay (2003) eq. (3.12), (3.20), (3.22) pp. 52-53 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay3e22 import bcoinbf


def test_bcoinbf_evidence_for_the_free_bias_is_the_beta_integral():
    fa, fb = 5, 3
    # (3.12): integral of p^Fa (1-p)^Fb dp = Fa! Fb! / (Fa+Fb+1)!
    exact = math.factorial(fa) * math.factorial(fb) / math.factorial(fa + fb + 1)
    res = bcoinbf(fa, fb)
    assert res["evidence1"] == pytest.approx(exact, rel=1e-12)


def test_bcoinbf_evidence_for_the_fixed_bias_is_the_plain_likelihood():
    fa, fb, p0 = 4, 6, 1.0 / 6.0
    res = bcoinbf(fa, fb, p0)
    assert res["evidence0"] == pytest.approx(p0 ** fa * (1.0 - p0) ** fb, rel=1e-12)
    assert res["ratio"] == pytest.approx(res["evidence1"] / res["evidence0"], rel=1e-9)


def test_bcoinbf_prefers_the_fixed_bias_when_the_data_match_it():
    # data generated at exactly p0 should not favour the free-bias model
    res = bcoinbf(1, 5, 1.0 / 6.0)
    assert res["ratio"] < 3.0
