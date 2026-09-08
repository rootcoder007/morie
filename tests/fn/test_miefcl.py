"""Tests for miefcl (Rubin's rules for multiple imputation).

Replaces the generated stub, which imported
``multiple_imputation_combine``.
"""

import math

from morie.fn.miefcl import miefcl


def test_rubins_rules_by_hand():
    est = [1.0, 1.4, 0.8, 1.2, 1.1]
    var = [0.04, 0.05, 0.03, 0.045, 0.04]
    res = miefcl(est, var)
    m = len(est)
    qbar = sum(est) / m
    ubar = sum(var) / m
    b = sum((e - qbar) ** 2 for e in est) / (m - 1)
    total = ubar + (1.0 + 1.0 / m) * b
    assert abs(res["estimate"] - qbar) < 1e-12
    assert abs(res["ubar"] - ubar) < 1e-12
    assert abs(res["b"] - b) < 1e-12
    assert abs(res["t"] - total) < 1e-12
    assert abs(res["se"] - math.sqrt(total)) < 1e-12


def test_the_relative_increase_and_fraction_missing():
    est = [1.0, 1.4, 0.8, 1.2, 1.1]
    var = [0.04, 0.05, 0.03, 0.045, 0.04]
    res = miefcl(est, var)
    m = len(est)
    riv = (1.0 + 1.0 / m) * res["b"] / res["ubar"]
    assert abs(res["riv"] - riv) < 1e-12
    assert abs(res["lambda_"] - riv / (1.0 + riv)) < 1e-12
    assert 0.0 < res["fmi"] < 1.0


def test_identical_imputations_leave_no_between_variance():
    res = miefcl([2.0] * 4, [0.1] * 4)
    assert abs(res["b"]) < 1e-15
    assert abs(res["t"] - 0.1) < 1e-12
    assert abs(res["riv"]) < 1e-15


def test_more_disagreement_inflates_the_standard_error():
    tight = miefcl([1.0, 1.01, 0.99, 1.0], [0.05] * 4)["se"]
    loose = miefcl([1.0, 2.0, 0.0, 1.5], [0.05] * 4)["se"]
    assert loose > tight


def test_the_barnard_rubin_df_is_used_when_the_complete_df_is_given():
    est = [1.0, 1.4, 0.8, 1.2]
    var = [0.04, 0.05, 0.03, 0.045]
    small = miefcl(est, var, nu_com=10)["df"]
    large = miefcl(est, var, nu_com=10000)["df"]
    assert small < large


def test_validation():
    for call in (lambda: miefcl([1.0], [0.1]),
                 lambda: miefcl([1.0, 2.0], [0.1]),
                 lambda: miefcl([1.0, 2.0], [-0.1, 0.1]),
                 lambda: miefcl([1.0, 1.0], [0.0, 0.0])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
