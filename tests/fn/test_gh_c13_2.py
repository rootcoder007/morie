"""Verification tests for gh_c13_2.

Ghosal and van der Vaart (2017), sec. 13.2, the Dirichlet-process posterior survival function.
"""

import math

import pytest

from morie.fn.gh_c13_2 import ghosal_surv_dp_km


def test_a_vanishing_concentration_recovers_kaplan_meier():
    # sec. 13.2: alpha -> 0 gives the Kaplan-Meier estimator exactly
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = ghosal_surv_dp_km(x, alpha=1e-8)
    dp = [float(v) for v in res["survival_dp"]]
    km = [float(v) for v in res["survival_km"]]
    for a, b in zip(dp, km):
        assert a == pytest.approx(b, abs=1e-6)
    assert res["max_abs_diff_to_km"] < 1e-6


def test_the_posterior_survival_is_a_decreasing_function_bounded_by_one():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = ghosal_surv_dp_km(x, alpha=1.0)
    dp = [float(v) for v in res["survival_dp"]]
    assert all(0.0 <= v <= 1.0 for v in dp)
    assert all(b <= a + 1e-12 for a, b in zip(dp, dp[1:]))


def test_more_prior_concentration_pulls_away_from_kaplan_meier():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    weak = ghosal_surv_dp_km(x, alpha=0.01)["max_abs_diff_to_km"]
    strong = ghosal_surv_dp_km(x, alpha=10.0)["max_abs_diff_to_km"]
    assert strong > weak
