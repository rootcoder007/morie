"""ordct: linear-by-linear association for ordered categories.

Gibbons & Chakraborti 5e, Ch 14 (Analysis of Count Data).
"""

import numpy as np
import pytest

from morie.fn.ordct import ordered_categories as oc


def _expand(tab, rs, cs):
    """Expand a contingency table into paired scores, one pair per count."""
    a, b = [], []
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            a += [rs[i]] * int(tab[i, j])
            b += [cs[j]] * int(tab[i, j])
    return np.array(a, float), np.array(b, float)


def test_ordct_statistic_is_n_minus_one_times_r_squared():
    """The linear-by-linear statistic M^2 = (N-1) r^2, with r the Pearson
    correlation of the row and column scores over the expanded table."""
    tab = np.array([[10.0, 5.0, 2.0], [5.0, 10.0, 5.0], [2.0, 5.0, 10.0]])
    rs = cs = np.array([1.0, 2.0, 3.0])
    a, b = _expand(tab, rs, cs)
    r = float(np.corrcoef(a, b)[0, 1])
    got = oc(tab, row_scores=rs, col_scores=cs)
    assert got["correlation"] == pytest.approx(r)
    assert got["statistic"] == pytest.approx((a.size - 1) * r**2)
    assert got["df"] == 1


def test_ordct_detects_a_monotone_trend_that_chi_square_would_dilute():
    """A clean diagonal trend gives a large statistic and a tiny p."""
    tab = np.array([[30.0, 5.0, 1.0], [5.0, 30.0, 5.0], [1.0, 5.0, 30.0]])
    r = oc(tab)
    assert r["correlation"] > 0.6
    assert r["p_value"] < 1e-10


def test_ordct_no_association_gives_a_statistic_near_zero():
    tab = np.outer([1.0, 1.0, 1.0], [20.0, 20.0, 20.0])
    r = oc(tab)
    assert r["correlation"] == pytest.approx(0.0, abs=1e-9)
    assert r["statistic"] == pytest.approx(0.0, abs=1e-9)
    assert r["p_value"] == pytest.approx(1.0)


def test_ordct_reversing_the_column_scores_flips_the_sign_only():
    """The test is about a LINEAR trend, so reversing one ordering negates r
    and leaves the squared statistic alone."""
    tab = np.array([[20.0, 6.0, 2.0], [6.0, 20.0, 6.0], [2.0, 6.0, 20.0]])
    a = oc(tab, col_scores=np.array([1.0, 2.0, 3.0]))
    b = oc(tab, col_scores=np.array([3.0, 2.0, 1.0]))
    assert a["correlation"] == pytest.approx(-b["correlation"])
    assert a["statistic"] == pytest.approx(b["statistic"])


def test_ordct_default_scores_are_the_equally_spaced_integers():
    tab = np.array([[12.0, 4.0], [4.0, 12.0]])
    assert oc(tab)["correlation"] == pytest.approx(
        oc(tab, row_scores=np.array([1.0, 2.0]), col_scores=np.array([1.0, 2.0]))["correlation"]
    )


def test_ordct_is_invariant_to_affine_rescaling_of_the_scores():
    """Only the spacing pattern matters, not the units."""
    tab = np.array([[15.0, 5.0, 2.0], [5.0, 15.0, 5.0], [2.0, 5.0, 15.0]])
    s = np.array([1.0, 2.0, 3.0])
    a = oc(tab, row_scores=s, col_scores=s)
    b = oc(tab, row_scores=10 * s + 7, col_scores=10 * s + 7)
    assert a["statistic"] == pytest.approx(b["statistic"])
