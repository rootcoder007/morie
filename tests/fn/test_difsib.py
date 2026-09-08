"""Tests for difsib.dif_sibtest (SIBTEST DIF, Shealy and Stout 1993).

Anchored on a hand-designed two-level fixture where every quantity is
exact:

    level 0: ref 1,1,0,0 -> Ybar .50, s2 1/3, n 4
             foc 1,0,0,0 -> Ybar .25, s2 1/4, n 4
    level 1: ref 1,1,1,0 -> Ybar .75, s2 1/4, n 4
             foc 1,1,0,0 -> Ybar .50, s2 1/3, n 4

    pstar = (1/2, 1/2)
    beta  = .5(.50-.25) + .5(.75-.50) = 0.25
    var   = 2 * (1/2)^2 * (1/4 + 1/3)/4
"""

import math

import pytest

from morie.fn.difsib import dif_sibtest

Y = [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0]
G = ["r"] * 4 + ["f"] * 4 + ["r"] * 4 + ["f"] * 4
M = [0] * 8 + [1] * 8


def test_hand_computed_beta_and_sigma():
    res = dif_sibtest(Y, G, matching=M)
    assert res["n_levels"] == 2
    assert res["beta"] == pytest.approx(0.25, abs=1e-15)
    sigma = math.sqrt(2 * 0.25 * (1 / 4 + 1 / 3) / 4)
    assert res["sigma"] == pytest.approx(sigma, abs=1e-15)
    assert res["statistic"] == pytest.approx((0.25 / sigma) ** 2, rel=1e-12)
    assert res["df"] == 1


def test_naming_the_other_reference_negates_beta():
    a = dif_sibtest(Y, G, matching=M)
    b = dif_sibtest(Y, G, matching=M, reference="f")
    assert b["beta"] == pytest.approx(-a["beta"], abs=1e-15)
    assert b["statistic"] == pytest.approx(a["statistic"], rel=1e-12)


def test_bare_relabelling_does_not_flip_the_sign():
    """The default reference is the first value encountered, so swapping
    the two labels leaves the same examinees in the reference role.  That
    order-dependence is documented and is why reference= exists."""
    a = dif_sibtest(Y, G, matching=M)
    b = dif_sibtest(Y, ["f" if g == "r" else "r" for g in G], matching=M)
    assert b["beta"] == pytest.approx(a["beta"], abs=1e-15)


def test_no_dif_gives_zero_beta():
    res = dif_sibtest([1, 1, 0, 0] * 4, G, matching=M)
    assert res["beta"] == pytest.approx(0.0, abs=1e-15)
    assert res["p_value"] > 0.99


def test_pstar_is_a_probability_vector():
    res = dif_sibtest(Y, G, matching=M)
    assert sum(float(v) for v in res["pstar"]) == pytest.approx(1.0)


def test_studied_alias_selects_the_suspect_item():
    a = dif_sibtest(Y, G, matching=M)
    b = dif_sibtest([0] * 16, G, studied=Y, matching=M)
    assert b["beta"] == pytest.approx(a["beta"], abs=1e-15)


def test_matching_is_required():
    with pytest.raises(ValueError):
        dif_sibtest(Y, G)


def test_levels_without_variance_are_dropped():
    """A level where one group is constant contributes nothing to sigma;
    mirt drops it and renormalises pstar, and so must this."""
    y = [1, 1, 0, 0, 1, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1, 1]
    res = dif_sibtest(y, G, matching=M)
    assert res["n_levels"] == 1
    assert list(res["levels"]) == [0.0]
    assert sum(float(v) for v in res["pstar"]) == pytest.approx(1.0)
