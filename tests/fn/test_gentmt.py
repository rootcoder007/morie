"""Tests for gentmt -- IP-weighted marginal structural models.

Replaces a generated test that called a stub returning
mean(y). Fixtures and the computed truth live in
_msm_fixture.py; the full anchor is
ledger/wave3/anchor_msm4.py.
"""

import pytest

from morie.fn.gentmt import generalized_treatment_msm

from ._msm_fixture import N, TH1, dose, feedback  # noqa: F401



def test_gentmt_three_routes_agree_on_the_dose_slope(dose):
    w = generalized_treatment_msm(dose["Y"], dose["D"], dose["X"])
    assert w["estimate"] == pytest.approx(1.5, abs=0.12)
    assert abs(w["crude"] - 1.5) > 0.1
    assert w["finite_variance"] is True

    sub = generalized_treatment_msm(dose["Y"], dose["D"], dose["X"],
                                    method="subclassify", n_strata=5)
    assert sub["estimate"] == pytest.approx(1.5, abs=0.2)

    dr = generalized_treatment_msm(dose["Y"], dose["D"], dose["X"],
                                   method="doseresponse")
    assert dr["estimate"] == pytest.approx(1.5, abs=0.3)
    assert all(dr["curve"][t + 1] > dr["curve"][t]
               for t in range(len(dr["curve"]) - 1))


def test_gentmt_refuses_a_binary_exposure(feedback):
    with pytest.raises(ValueError):
        generalized_treatment_msm(feedback["Y"], feedback["A"][0],
                                  feedback["L"][0])


def test_gentmt_argument_checks(dose):
    with pytest.raises(ValueError):
        generalized_treatment_msm(dose["Y"], dose["D"], dose["X"],
                                  method="nope")
    with pytest.raises(ValueError):
        generalized_treatment_msm(dose["Y"], dose["D"], dose["X"],
                                  degree=0)
    with pytest.raises(ValueError):
        generalized_treatment_msm(dose["Y"], dose["D"], dose["X"],
                                  method="subclassify", n_strata=1)
