"""Verification tests for km108.

Kamath, Keenan, Somers and Sorenson (2024), eq. 6.32, the differential-privacy guarantee check. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km108 import kamath_ch6_differential_privacy


def test_the_privacy_check_compares_the_output_ratio_to_the_budget():
    # Eq 6.32: P[M(A) in S] <= e^eps P[M(B) in S]
    M = lambda D: ({"o1": 0.6, "o2": 0.4} if D == "A"
                   else {"o1": 0.3, "o2": 0.7})
    res = kamath_ch6_differential_privacy(M, "A", "B", ["o1"], 1.0)
    assert res["satisfied"] is True
    # 0.6 / 0.3 = 2, so the required budget is log 2
    assert res["epsilon_required"] == pytest.approx(math.log(2.0), rel=1e-12)


def test_a_budget_below_the_requirement_fails_the_check():
    M = lambda D: ({"o1": 0.6, "o2": 0.4} if D == "A"
                   else {"o1": 0.3, "o2": 0.7})
    assert kamath_ch6_differential_privacy(M, "A", "B", ["o1"], 0.5)["satisfied"] is False


def test_identical_mechanisms_need_no_budget_at_all():
    M = lambda D: {"o1": 0.5, "o2": 0.5}
    res = kamath_ch6_differential_privacy(M, "A", "B", ["o1"], 0.0)
    assert res["epsilon_required"] == pytest.approx(0.0, abs=1e-12)
    assert res["satisfied"] is True
