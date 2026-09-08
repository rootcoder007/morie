"""Tests for lggvls -- IP-weighted marginal structural models.

Replaces a generated test that called a stub returning
mean(y). Fixtures and the computed truth live in
_msm_fixture.py; the full anchor is
ledger/wave3/anchor_msm4.py.
"""

import pytest

from morie.fn.lggvls import laggedval_iptw
from morie.fn.tdcvar import time_dep_covariate

from ._msm_fixture import N, TH1, dose, feedback  # noqa: F401



def test_lggvls_matches_the_full_history_and_the_lag_bites(feedback):
    r = time_dep_covariate(feedback["Y"], feedback["A"], feedback["L"])
    g = laggedval_iptw(feedback["Y"], feedback["A"], feedback["L"], lag=1)
    assert g["estimate"] == pytest.approx(r["estimate"], abs=0.03)
    assert g["estimate"] == pytest.approx(feedback["truth"], abs=0.12)
    # lag=0 drops L0 from the time-1 model, so the weights must differ
    g0 = laggedval_iptw(feedback["Y"], feedback["A"], feedback["L"], lag=0)
    assert max(abs(g0["weights"][i] - g["weights"][i])
               for i in range(N)) > 1e-6


def test_lggvls_contrasts_differ(feedback):
    cum = laggedval_iptw(feedback["Y"], feedback["A"], feedback["L"])
    ever = laggedval_iptw(feedback["Y"], feedback["A"], feedback["L"],
                          contrast="everexposed")
    assert abs(ever["estimate"] - cum["estimate"]) > 0.05


def test_lggvls_argument_checks(feedback):
    with pytest.raises(ValueError):
        laggedval_iptw(feedback["Y"], feedback["A"], feedback["L"], lag=-1)
    with pytest.raises(ValueError):
        laggedval_iptw(feedback["Y"], feedback["A"], feedback["L"],
                       contrast="nope")


# ----------------------------------------------------------- polkrn
