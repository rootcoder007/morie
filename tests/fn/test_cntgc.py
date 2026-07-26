"""cntgc: Pearson contingency coefficient (Gibbons & Chakraborti 5e, Ch 14)."""

import numpy as np
import pytest
from scipy import stats

from morie.fn.cntgc import contingency_coefficient as cc


def test_cntgc_matches_the_closed_form_on_a_hand_table():
    """C = sqrt(chi2 / (chi2 + n)), with chi2 from the usual Pearson test."""
    tab = np.array([[10.0, 20.0], [30.0, 40.0]])
    chi2 = float(stats.chi2_contingency(tab, correction=False)[0])
    n = tab.sum()
    r = cc(tab)
    assert r["chi2"] == pytest.approx(chi2)
    assert r["statistic"] == pytest.approx(np.sqrt(chi2 / (chi2 + n)))
    assert r["df"] == 1


def test_cntgc_cramers_v_uses_the_smaller_dimension():
    """V = sqrt(chi2 / (n * min(r-1, c-1)))."""
    tab = np.array([[10.0, 20.0, 30.0], [30.0, 40.0, 10.0]])
    chi2 = float(stats.chi2_contingency(tab, correction=False)[0])
    n = tab.sum()
    assert cc(tab)["cramers_v"] == pytest.approx(np.sqrt(chi2 / (n * min(1, 2))))


def test_cntgc_independence_gives_zero_association():
    """A table that is exactly the outer product of its margins has chi2 = 0."""
    tab = np.outer([2.0, 3.0], [4.0, 6.0]) * 5
    r = cc(tab)
    assert r["chi2"] == pytest.approx(0.0, abs=1e-9)
    assert r["statistic"] == pytest.approx(0.0, abs=1e-9)
    assert r["p_value"] == pytest.approx(1.0)


def test_cntgc_C_cannot_reach_one_and_max_C_says_so():
    """C is bounded by sqrt((k-1)/k) < 1 for a k x k table -- the reason
    max_C is returned at all, since C alone is not comparable across table
    sizes."""
    for k in (2, 3, 4):
        tab = np.eye(k) * 50.0
        r = cc(tab)
        assert r["max_C"] == pytest.approx(np.sqrt((k - 1) / k))
        assert r["statistic"] <= r["max_C"] + 1e-9
        assert r["statistic"] < 1.0


def test_cntgc_perfect_association_hits_the_bound():
    """A diagonal table is maximal association, so C should equal max_C."""
    r = cc(np.eye(3) * 40.0)
    assert r["statistic"] == pytest.approx(r["max_C"], rel=1e-9)


def test_cntgc_is_invariant_to_transposing_the_table():
    tab = np.array([[5.0, 15.0, 10.0], [20.0, 10.0, 25.0]])
    assert cc(tab)["statistic"] == pytest.approx(cc(tab.T)["statistic"])


def test_cntgc_is_exactly_invariant_to_multiplying_the_whole_table():
    """C depends on the PROPORTIONS only, not the sample size.

    chi2 is proportional to n at fixed proportions, so the n cancels in
    C = sqrt(chi2 / (chi2 + n)). Measured: C = 0.088736 at x1, x10 and x100.
    (chi2 itself is not invariant -- it goes 0.7937 -> 7.9365 -> 79.3651 --
    which is exactly why C is reported alongside it.)
    """
    tab = np.array([[10.0, 20.0], [30.0, 40.0]])
    base = cc(tab)
    for s in (10, 100):
        scaled = cc(tab * s)
        assert scaled["statistic"] == pytest.approx(base["statistic"], rel=1e-12)
        assert scaled["chi2"] == pytest.approx(base["chi2"] * s, rel=1e-12)
