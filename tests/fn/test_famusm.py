"""famusm: transmission disequilibrium test (Spielman, McGinnis & Ewens 1993)."""

from morie.fn import _array_core as np
import pytest
from morie.fn import _stats_core as stats

from morie.fn.famusm import family_based_assoc as tdt


def test_famusm_matches_the_chi_square_formula():
    """chi2 = (b-c)^2/(b+c) with 1 df."""
    r = tdt((30, 10))
    assert r["statistic"] == pytest.approx(400.0 / 40.0) == pytest.approx(10.0)
    assert r["df"] == 1
    assert r["p_value"] == pytest.approx(float(stats.chi2.sf(10.0, 1)))


def test_famusm_equal_transmission_is_the_null():
    """b = c means no distortion at all: chi2 = 0, p = 1."""
    r = tdt((25, 25))
    assert r["statistic"] == 0.0
    assert r["p_value"] == pytest.approx(1.0)


def test_famusm_is_symmetric_in_b_and_c():
    """The test is two-sided; swapping transmitted/untransmitted cannot change it."""
    assert tdt((30, 10))["statistic"] == pytest.approx(tdt((10, 30))["statistic"])


def test_famusm_agrees_with_mcnemar():
    """The TDT is McNemar's test on the transmission pair."""
    b, c = 42, 18
    expected = float(stats.chi2.sf((b - c) ** 2 / (b + c), 1))
    assert tdt((b, c))["p_value"] == pytest.approx(expected)


def test_famusm_accepts_per_trio_indicators():
    """An (n, 2) array of (transmitted, untransmitted) is summed to (b, c)."""
    trios = np.array([[1, 0]] * 30 + [[0, 1]] * 10, dtype=float)
    assert tdt(trios)["b"] == 30
    assert tdt(trios)["c"] == 10
    assert tdt(trios)["statistic"] == pytest.approx(tdt((30, 10))["statistic"])


def test_famusm_odds_ratio_is_b_over_c():
    assert tdt((30, 10))["odds_ratio"] == pytest.approx(3.0)
    assert np.isinf(tdt((5, 0))["odds_ratio"])


def test_famusm_reports_informative_count_for_the_asymptotic_caveat():
    """b + c is what tells the caller whether chi2_1 is safe to trust."""
    assert tdt((3, 1))["n_informative"] == 4


def test_famusm_no_informative_parents_is_undefined():
    """Homozygous parents transmit no information; b + c = 0 is not chi2 = 0."""
    with pytest.raises(ValueError, match="no heterozygous"):
        tdt((0, 0))


def test_famusm_rejects_negative_and_fractional_counts():
    with pytest.raises(ValueError, match="non-negative"):
        tdt((-1, 5))
    with pytest.raises(ValueError, match="whole numbers"):
        tdt((1.5, 5))
