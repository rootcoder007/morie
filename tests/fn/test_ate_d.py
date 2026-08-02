"""Tests for ate_d.ate_definition."""

from morie.fn import _array_core as np
import pytest
from morie.fn import _stats_core as stats

from morie.fn.ate_d import ate_definition


def test_ate_is_the_difference_in_means():
    rng = np.random.default_rng(0)
    y1, y0 = rng.normal(5, 1, 200), rng.normal(3, 1, 200)
    res = ate_definition(y1, y0)
    assert res["ate"] == pytest.approx(y1.mean() - y0.mean())


def test_paired_matches_a_paired_t_test():
    rng = np.random.default_rng(1)
    y0 = rng.normal(0, 1, 150)
    y1 = y0 + 2 + rng.normal(0, 0.3, 150)
    res = ate_definition(y1, y0, paired=True)
    ref = stats.ttest_rel(y1, y0)
    assert res["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-12)


def test_unpaired_matches_welch():
    rng = np.random.default_rng(2)
    y1, y0 = rng.normal(5, 2, 120), rng.normal(3, 1, 90)
    res = ate_definition(y1, y0, paired=False)
    ref = stats.ttest_ind(y1, y0, equal_var=False)
    assert res["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-12)


def test_pairing_shrinks_the_standard_error_when_arms_are_correlated():
    """The whole reason `paired` is explicit: it changes the SE a lot."""
    rng = np.random.default_rng(3)
    y0 = rng.normal(0, 5, 200)
    y1 = y0 + 1.0                      # perfectly correlated arms
    se_paired = ate_definition(y1, y0, paired=True)["se"]
    se_indep = ate_definition(y1, y0, paired=False)["se"]
    assert se_paired < se_indep / 10


def test_confidence_interval_brackets_the_estimate_and_respects_alpha():
    rng = np.random.default_rng(4)
    y1, y0 = rng.normal(1, 1, 100), rng.normal(0, 1, 100)
    r95 = ate_definition(y1, y0)
    r99 = ate_definition(y1, y0, alpha=0.01)
    assert r95["ci_low"] < r95["ate"] < r95["ci_high"]
    assert r99["ci_high"] - r99["ci_low"] > r95["ci_high"] - r95["ci_low"]


def test_zero_effect_is_not_detected():
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 300)
    assert ate_definition(y, y.copy())["ate"] == pytest.approx(0.0)


def test_validates_inputs():
    rng = np.random.default_rng(6)
    y1, y0 = rng.normal(0, 1, 50), rng.normal(0, 1, 40)
    with pytest.raises(ValueError, match="same units in both arms"):
        ate_definition(y1, y0, paired=True)
    with pytest.raises(ValueError, match="must be finite"):
        ate_definition(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match=r"alpha must lie in \(0, 1\)"):
        ate_definition(y1, y1, alpha=0)
    with pytest.raises(ValueError, match="must not be empty"):
        ate_definition(np.array([]), y0)
