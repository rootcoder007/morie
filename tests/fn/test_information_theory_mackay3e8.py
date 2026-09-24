"""Verification tests for information_theory_mackay3e8.bcoinlik.

The expected values are recomputed from MacKay (2003) eq. (3.8) p. 51 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay3e8 import bcoinlik


def test_bcoinlik_is_the_bernoulli_likelihood():
    pa, fa, fb = 0.3, 5, 7
    res = bcoinlik(pa, fa, fb)
    assert res["likelihood"] == pytest.approx(pa ** fa * (1.0 - pa) ** fb, rel=1e-12)
    assert res["loglik"] == pytest.approx(fa * math.log(pa) + fb * math.log1p(-pa), abs=1e-12)


def test_bcoinlik_is_maximised_at_the_sample_proportion():
    fa, fb = 6, 4
    best = max((bcoinlik(p / 1000.0, fa, fb)["likelihood"], p / 1000.0)
               for p in range(1, 1000))
    assert best[1] == pytest.approx(fa / (fa + fb), abs=2e-3)
