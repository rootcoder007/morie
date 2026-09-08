"""Tests for polkrn -- IP-weighted marginal structural models.

Replaces a generated test that called a stub returning
mean(y). Fixtures and the computed truth live in
_msm_fixture.py; the full anchor is
ledger/wave3/anchor_msm4.py.
"""

import pytest

from morie.fn.polkrn import polynomial_kernel_msm
from morie.fn.tdcvar import time_dep_covariate

from ._msm_fixture import N, TH1, dose, feedback  # noqa: F401



def test_polkrn_degree_one_is_the_linear_msm(feedback):
    r = time_dep_covariate(feedback["Y"], feedback["A"], feedback["L"])
    p = polynomial_kernel_msm(feedback["Y"], feedback["A"], feedback["L"],
                              degree=1, basis="polynomial")
    assert p["estimate"] == pytest.approx(r["estimate"], abs=1e-9)


def test_polkrn_finds_a_quadratic_dose_response(feedback):
    Yq = [feedback["Y"][i]
          + 0.8 * (feedback["A"][0][i] + feedback["A"][1][i]) ** 2
          for i in range(N)]
    p = polynomial_kernel_msm(Yq, feedback["A"], feedback["L"], degree=2,
                              basis="both")
    assert p["coef_polynomial"][2] == pytest.approx(0.8, abs=0.15)
    # the kernel basis should trace roughly the same curve
    gap = max(abs(p["curve_kernel"][t] - p["curve_polynomial"][t])
              for t in range(len(p["grid"])))
    assert gap < 0.6


def test_polkrn_summaries_and_checks(feedback):
    lin = polynomial_kernel_msm(feedback["Y"], feedback["A"],
                                feedback["L"], degree=1,
                                basis="polynomial")
    dur = polynomial_kernel_msm(feedback["Y"], feedback["A"],
                                feedback["L"], degree=1,
                                basis="polynomial", summary="duration")
    # for a binary treatment, duration and cumulative are the same thing
    assert dur["estimate"] == pytest.approx(lin["estimate"], abs=1e-9)
    with pytest.raises(ValueError):
        polynomial_kernel_msm(feedback["Y"], feedback["A"],
                              feedback["L"], degree=0)
    with pytest.raises(ValueError):
        polynomial_kernel_msm(feedback["Y"], feedback["A"],
                              feedback["L"], basis="nope")


# ----------------------------------------------------------- gentmt
