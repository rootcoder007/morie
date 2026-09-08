"""Tests for tdcvar -- IP-weighted marginal structural models.

Replaces a generated test that called a stub returning
mean(y). Fixtures and the computed truth live in
_msm_fixture.py; the full anchor is
ledger/wave3/anchor_msm4.py.
"""

import pytest

from morie.fn.tdcvar import time_dep_covariate

from ._msm_fixture import N, TH1, dose, feedback  # noqa: F401



def test_the_fixture_is_additive(feedback):
    """If the regimes were not additive the cumulative MSM would be
    misspecified and 'recovering the truth' would be meaningless."""
    ey = feedback["EY"]
    a0_only = ey[(1.0, 0.0)] - ey[(0.0, 0.0)]
    a1_only = ey[(0.0, 1.0)] - ey[(0.0, 0.0)]
    assert a0_only == pytest.approx(a1_only, abs=0.02)
    assert feedback["truth"] == pytest.approx(TH1, abs=0.02)


# ----------------------------------------------------------- tdcvar


def test_tdcvar_recovers_the_effect_both_naive_fixes_miss(feedback):
    r = time_dep_covariate(feedback["Y"], feedback["A"], feedback["L"])
    truth = feedback["truth"]
    assert r["estimate"] == pytest.approx(truth, abs=0.12)
    assert abs(r["unadjusted"] - truth) > 0.1
    assert abs(r["adjusted"] - truth) > 0.1
    # Ch. 20: over-adjustment and confounding pull opposite ways
    assert (r["adjusted"] - truth) * (r["unadjusted"] - truth) < 0.0
    assert r["mean_weight"] == pytest.approx(1.0, abs=0.05)


def test_tdcvar_argument_checks(feedback):
    with pytest.raises(ValueError):
        time_dep_covariate(feedback["Y"], feedback["A"],
                           [feedback["L"][0]])
    with pytest.raises(ValueError):
        time_dep_covariate(feedback["Y"], feedback["A"], feedback["L"],
                           contrast="nope")


# ----------------------------------------------------------- lggvls
