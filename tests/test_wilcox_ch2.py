"""Wilcox (2017) chapter 2 robust statistics: the printed worked examples.

Each assertion is an anchor taken from the book, not from the code under
test, so a regression that keeps Python and R agreeing with each other
still fails here.
"""

import pytest

from morie.fn.bimid import bimid
from morie.fn.idealf import idealf
from morie.fn.outmad import outmad
from morie.fn.outms import outms

# p.27 worked example
_P27 = [-29.6, -20.9, -19.7, -15.4, -12.3, -8.0,
        -4.3, 0.8, 2.0, 6.2, 11.2, 25.0]
# p.32 masking demonstration
_P32 = [2] * 5 + [3] * 5 + [4] * 5 + [1000]
# p.33 MAD-median example
_P33 = [2, 2, 3, 3, 3, 4, 4, 4, 100000, 100000]


def test_idealf_matches_p27_worked_example():
    r = dict(idealf(_P27))
    assert r["j"] == 3
    assert r["h"] == pytest.approx(0.41667, abs=5e-6)
    assert r["q1"] == pytest.approx(-17.9, abs=0.05)


def test_idealf_upper_fourth_mirrors_the_lower():
    """q2 interpolates X[k] -> X[k-1]; on a symmetric sample q2 = -q1."""
    x = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, -7.0, 7.0]
    r = dict(idealf(x))
    assert r["q2"] == pytest.approx(-r["q1"], abs=1e-12)


def test_bimid_matches_closed_form_on_minus_one_zero_one():
    """M = 0, MAD = 1, Y = (-1/9, 0, 1/9), every a_i = 1."""
    from math import sqrt
    num = sqrt(3.0) * sqrt(2.0 * (80 / 81) ** 4)
    den = 1.0 + 2.0 * (80 / 81) * (1 - 5 * (1 / 81))
    assert dict(bimid([-1.0, 0.0, 1.0]))["zeta"] == pytest.approx(
        num / den, rel=0, abs=1e-15)


def test_bimid_rejects_a_sample_with_zero_mad():
    with pytest.raises(ValueError, match="median absolute deviation"):
        bimid([5.0, 5.0, 5.0, 5.0, 9.0])


def test_outms_detects_then_is_masked():
    """p.32: 1000 is flagged; appending 10000 masks it."""
    r = dict(outms(_P32))
    assert r["flag"][-1] == 1
    assert r["dis"][-1] == pytest.approx(3.75, abs=0.01)

    r2 = dict(outms(_P32 + [10000]))
    assert r2["flag"][_P32.index(1000)] == 0
    assert r2["dis"][15] == pytest.approx(0.14, abs=0.01)


def test_outmad_survives_the_masking_that_defeats_outms():
    """p.33: the MAD-median rule finds both extremes, mean/SD finds none."""
    assert dict(outmad(_P33))["n_out"] == 2
    assert dict(outms(_P33))["n_out"] == 0
    assert dict(outmad(_P33))["center"] == pytest.approx(3.5, abs=1e-12)
    assert dict(outmad(_P33))["scale"] == pytest.approx(0.7413, abs=5e-5)


def test_outmad_confusion_matrix():
    clean = [10.0, 10.1, 9.9, 10.2, 9.8, 10.05, 9.95, 10.15, 9.85, 10.3, 9.7]
    dirty = clean + [500.0, -400.0, 900.0]
    assert dict(outmad(clean))["n_out"] == 0            # no false positives
    flags = dict(outmad(dirty))["flag"]
    assert sum(flags[len(clean):]) == 3                 # all three found


def test_outmad_crit_exposes_the_hampel_identifier():
    """Hampel used 3.5, not Rousseeuw-van Zomeren's 2.24."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 20.0]
    assert dict(outmad(x, crit=2.24))["n_out"] >= dict(outmad(x, crit=3.5))["n_out"]


@pytest.mark.parametrize("fn", [idealf, bimid, outms, outmad])
def test_missing_values_are_rejected(fn):
    with pytest.raises(ValueError, match="missing value"):
        fn([1.0, 2.0, float("nan"), 4.0, 5.0])
