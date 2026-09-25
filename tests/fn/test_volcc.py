"""Tests for volcc.vol_christoffersen_cc."""

import math

import pytest

from morie.fn import _stats_core as stats
from morie.fn.volcc import vol_christoffersen_cc
from morie.fn.volkupiec import vol_kupiec_var_test

# t = 12, and the eleven transitions are hand-countable:
# 0->1 1->1 1->0 0->0 0->0 0->1 1->0 0->1 1->1 1->0 0->0
HITS = [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0]
N00, N01, N10, N11 = 3, 3, 3, 2


def _t(k, q):
    return k * math.log(q) if k else 0.0


def _lr_ind(n00, n01, n10, n11):
    """The docstring's first-order chain against the i.i.d. chain."""
    total = n00 + n01 + n10 + n11
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)
    pi = (n01 + n11) / total
    ll_null = _t(n00 + n10, 1.0 - pi) + _t(n01 + n11, pi)
    ll_alt = _t(n00, 1.0 - pi0) + _t(n01, pi0) + _t(n10, 1.0 - pi1) + _t(n11, pi1)
    return -2.0 * (ll_null - ll_alt)


def test_volcc_counts_the_transitions_and_adds_the_two_halves():
    out = vol_christoffersen_cc(HITS, 0.05)
    assert (out["n00"], out["n01"], out["n10"], out["n11"]) == (N00, N01, N10, N11)
    assert out["n00"] + out["n01"] + out["n10"] + out["n11"] == len(HITS) - 1
    assert out["n_obs"] == 12
    assert out["n_exceedances"] == 5
    assert out["pi01"] == pytest.approx(N01 / (N00 + N01), rel=1e-12)
    assert out["pi11"] == pytest.approx(N11 / (N10 + N11), rel=1e-12)
    assert out["pi"] == pytest.approx((N01 + N11) / 11, rel=1e-12)

    want_ind = _lr_ind(N00, N01, N10, N11)
    assert out["lr_ind"] == pytest.approx(want_ind, rel=1e-12)
    # LR_uc is Kupiec's statistic on the same sequence, unchanged
    want_uc = float(vol_kupiec_var_test(HITS, 0.05)["statistic"])
    assert out["lr_uc"] == pytest.approx(want_uc, rel=1e-12)
    # LR_cc = LR_uc + LR_ind, chi2_2
    assert out["statistic"] == pytest.approx(want_uc + want_ind, rel=1e-12)
    assert out["df"] == 2
    assert out["pvalue"] == pytest.approx(
        float(stats.chi2.sf(want_uc + want_ind, 2)), rel=1e-12)
    assert out["pvalue_uc"] == pytest.approx(float(stats.chi2.sf(want_uc, 1)), rel=1e-12)
    assert out["pvalue_ind"] == pytest.approx(
        float(stats.chi2.sf(want_ind, 1)), rel=1e-12)


def test_volcc_independence_half_vanishes_on_a_degenerate_chain():
    """All-zero hits give pi0 = pi1 = 0, so LR_ind is exactly zero and
    LR_cc collapses onto Kupiec's LR_uc."""
    out = vol_christoffersen_cc([0] * 40, 0.05)
    assert (out["n00"], out["n01"], out["n10"], out["n11"]) == (39, 0, 0, 0)
    assert out["lr_ind"] == pytest.approx(0.0, abs=1e-12)
    assert out["pvalue_ind"] == pytest.approx(1.0, rel=1e-12)
    assert out["lr_uc"] == pytest.approx(-2.0 * 40 * math.log(0.95), rel=1e-12)
    assert out["statistic"] == pytest.approx(out["lr_uc"], rel=1e-12)


def test_volcc_sees_clustering_that_kupiec_cannot():
    """Same breach count, same LR_uc; only the independence half moves.
    This is the failure mode the docstring says Kupiec alone misses."""
    spread = ([1] + [0] * 9) * 10
    clustered = [1] * 10 + [0] * 90
    a = vol_christoffersen_cc(spread, 0.10)
    b = vol_christoffersen_cc(clustered, 0.10)
    assert a["lr_uc"] == pytest.approx(b["lr_uc"], rel=1e-12)
    assert a["lr_uc"] == pytest.approx(0.0, abs=1e-12)  # rate == alpha exactly
    assert b["lr_ind"] > a["lr_ind"]
    assert b["pvalue_ind"] < 0.01
    # clustered breaches: nine of the ten are followed by another breach
    assert (b["n00"], b["n01"], b["n10"], b["n11"]) == (89, 0, 1, 9)


def test_volcc_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 2"):
        vol_christoffersen_cc([1])
    with pytest.raises(ValueError, match="0/1"):
        vol_christoffersen_cc([0, 2, 1])
