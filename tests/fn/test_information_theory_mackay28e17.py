"""Verification tests for information_theory_mackay28e17.mdlpost.

The expected values are recomputed from MacKay (2003) eq. (28.16)-(28.17) p. 352 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay28e17 import mdlpost


def test_mdlpost_is_the_sum_of_the_two_message_parts():
    ph, pdh, dd = 0.25, 0.5, 2.0
    res = mdlpost(ph, pdh, dd)
    assert res["model"] == pytest.approx(2.0, abs=1e-12)
    assert res["data"] == pytest.approx(-math.log2(pdh * dd), abs=1e-12)
    assert res["total"] == pytest.approx(res["model"] + res["data"], abs=1e-12)


def test_mdlpost_charges_more_for_a_less_probable_model():
    cheap = mdlpost(0.5, 0.5)["model"]
    dear = mdlpost(0.01, 0.5)["model"]
    assert dear > cheap
