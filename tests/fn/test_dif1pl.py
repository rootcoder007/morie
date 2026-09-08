"""Tests for dif1pl.dif_mantel_haenszel.

Anchored on a hand-computed 2 x 2 table.  With A=4, B=1, C=1, D=4 and
T=10:

    E = n1 m1 / T = 5*5/10 = 2.5
    V = n1 n2 m1 m0 / (T^2 (T-1)) = 625/900
    chi2 = (|4 - 2.5| - 0.5)^2 / V = 1 / (625/900) = 1.44
    alpha_MH = (A D / T) / (B C / T) = 16/1 = 16

The R arm additionally cross-checks against base R
``stats::mantelhaen.test``, which is a wholly separate implementation.
"""

import pytest

from morie.fn.dif1pl import dif_mantel_haenszel

Y1 = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
G1 = ["r"] * 5 + ["f"] * 5


def test_hand_computed_single_table():
    res = dif_mantel_haenszel(Y1, G1)
    assert res["sum_A"] == 4.0
    assert res["sum_E"] == pytest.approx(2.5)
    assert res["sum_V"] == pytest.approx(625.0 / 900.0)
    assert res["statistic"] == pytest.approx(1.44)
    assert res["alpha_MH"] == pytest.approx(16.0)
    assert res["df"] == 1


def test_balanced_table_has_zero_statistic():
    res = dif_mantel_haenszel([1, 1, 0, 0, 1, 1, 0, 0], ["r"] * 4 + ["f"] * 4)
    assert res["statistic"] == pytest.approx(0.0, abs=1e-15)
    assert res["alpha_MH"] == pytest.approx(1.0)
    assert res["delta_MH"] == pytest.approx(0.0, abs=1e-12)


def test_reference_choice_inverts_the_odds_ratio_not_the_chi_square():
    a = dif_mantel_haenszel(Y1, G1)
    b = dif_mantel_haenszel(Y1, G1, reference="f")
    assert b["statistic"] == pytest.approx(a["statistic"])
    assert b["alpha_MH"] == pytest.approx(1.0 / a["alpha_MH"])
    assert b["delta_MH"] == pytest.approx(-a["delta_MH"])


def test_uninformative_strata_are_dropped():
    """A stratum where everyone answers alike carries no information: its
    variance term is zero and it must not be counted."""
    y = Y1 + [1, 1, 1, 1, 1, 1]
    g = G1 + ["r"] * 3 + ["f"] * 3
    s = [1] * 10 + [2] * 6
    res = dif_mantel_haenszel(y, g, s)
    assert res["n_strata"] == 1
    assert res["statistic"] == pytest.approx(1.44)


def test_continuity_correction_can_be_turned_off():
    on = dif_mantel_haenszel(Y1, G1, correct=True)
    off = dif_mantel_haenszel(Y1, G1, correct=False)
    assert off["statistic"] > on["statistic"]
    assert off["statistic"] == pytest.approx(1.5**2 / (625.0 / 900.0))


def test_rejects_non_binary_and_wrong_group_count():
    with pytest.raises(ValueError):
        dif_mantel_haenszel([0, 1, 2, 1], ["r", "r", "f", "f"])
    with pytest.raises(ValueError):
        dif_mantel_haenszel([0, 1, 0, 1], ["r", "f", "x", "f"])
