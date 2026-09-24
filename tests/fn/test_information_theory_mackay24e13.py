"""Verification tests for information_theory_mackay24e13.sigevid.

The expected values are recomputed from MacKay (2003) eq. (24.13) p. 320 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay24e13 import sigevid


def test_sigevid_splits_into_best_fit_and_occam_factor():
    s, n, sigma, sigmamu = 4.0, 9, 1.5, 2.0
    bestfit = -n * math.log(math.sqrt(2.0 * math.pi) * sigma) - s / (2.0 * sigma ** 2)
    occam = math.log(math.sqrt(2.0 * math.pi) * sigma / math.sqrt(n) / sigmamu)
    res = sigevid(s, n, sigma, sigmamu)
    assert res["bestfit"] == pytest.approx(bestfit, abs=1e-12)
    assert res["logoccam"] == pytest.approx(occam, abs=1e-12)
    assert res["logevidence"] == pytest.approx(bestfit + occam, abs=1e-12)


def test_sigevid_occam_factor_penalises_a_wider_prior():
    narrow = sigevid(4.0, 9, 1.5, 0.5)["logoccam"]
    wide = sigevid(4.0, 9, 1.5, 5.0)["logoccam"]
    assert wide < narrow
