"""cncrd: Kendall's coefficient of concordance W for incomplete rankings.

Gibbons & Chakraborti, *Nonparametric Statistical Inference*, 5th ed.,
section 12.5 "The Coefficient of Concordance for k Sets of Incomplete
Rankings" -- verified in the PDF (the section heading appears verbatim).
"""

import numpy as np
import pytest

from morie.fn.cncrd import concordance_incomplete as W


def test_cncrd_perfect_agreement_gives_one():
    """Every ranker orders the objects identically -> W = 1."""
    col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert W(np.column_stack([col] * 4))["statistic"] == pytest.approx(1.0)


def test_cncrd_perfect_reversal_between_two_rankers_gives_zero():
    """Two rankers in exact opposition cancel: every rank sum is equal, so
    there is no concordance at all."""
    col = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.column_stack([col, col[::-1]])
    assert W(x)["statistic"] == pytest.approx(0.0, abs=1e-12)


def test_cncrd_matches_the_closed_form_on_complete_rankings():
    """W = 12S / (k^2 (n^3 - n)) with S the sum of squared deviations of the
    object rank sums from their mean."""
    x = np.array(
        [
            [1.0, 1.0, 2.0],
            [2.0, 3.0, 1.0],
            [3.0, 2.0, 4.0],
            [4.0, 4.0, 3.0],
        ]
    )
    n, k = x.shape
    rank_sums = x.sum(axis=1)
    S = float(((rank_sums - rank_sums.mean()) ** 2).sum())
    expected = 12 * S / (k**2 * (n**3 - n))
    assert W(x)["statistic"] == pytest.approx(expected)


def test_cncrd_chi_square_is_k_times_n_minus_one_times_W():
    x = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 2.0], [4.0, 4.0]])
    n, k = x.shape
    r = W(x)
    assert r["df"] == n - 1
    assert r["chi2"] == pytest.approx(k * (n - 1) * r["statistic"])


def test_cncrd_lies_in_the_unit_interval():
    rng = np.random.default_rng(1401)
    for _ in range(30):
        n, k = 6, 4
        x = np.column_stack([rng.permutation(n) + 1.0 for _ in range(k)])
        w = W(x)["statistic"]
        assert 0.0 <= w <= 1.0


def test_cncrd_tolerates_gaps_which_is_the_whole_point_of_12_5():
    """Section 12.5 is specifically the INCOMPLETE-rankings case: NaN marks
    an object a ranker did not rank, and must not be read as a rank."""
    x = np.array(
        [
            [1.0, 1.0, np.nan],
            [2.0, 2.0, 1.0],
            [3.0, np.nan, 2.0],
            [4.0, 3.0, 3.0],
        ]
    )
    r = W(x)
    assert np.isfinite(r["statistic"])
    assert 0.0 <= r["statistic"] <= 1.0
    assert r["statistic"] > 0.5, "these rankers largely agree"


def test_cncrd_reports_its_shape():
    r = W(np.column_stack([np.arange(1.0, 6.0)] * 3))
    assert r["n"] == 5 and r["k"] == 3
