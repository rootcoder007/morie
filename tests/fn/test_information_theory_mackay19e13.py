"""Verification tests for information_theory_mackay19e13.sexdfdt.

The expected values are recomputed from MacKay (2003) eq. (19.13) p. 273 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay19e13 import sexdfdt


def test_sexdfdt_is_eta_times_the_root_of_f_one_minus_f_times_g():
    f, g, eta = 0.3, 100.0, 0.6
    res = sexdfdt(f, g, eta)
    assert res["dfbardt"] == pytest.approx(eta * math.sqrt(f * (1.0 - f) * g), rel=1e-12)


def test_sexdfdt_growth_is_fastest_at_one_half_and_zero_at_the_extremes():
    g = 50.0
    assert sexdfdt(0.0, g)["dfbardt"] == pytest.approx(0.0, abs=1e-12)
    assert sexdfdt(1.0, g)["dfbardt"] == pytest.approx(0.0, abs=1e-12)
    mid = sexdfdt(0.5, g)["dfbardt"]
    assert mid > sexdfdt(0.2, g)["dfbardt"] > 0.0


def test_sexdfdt_default_eta_is_the_books_constant():
    res = sexdfdt(0.5, 1.0)
    assert res["eta"] == pytest.approx(math.sqrt(2.0 / (math.pi + 2.0)), rel=1e-12)
