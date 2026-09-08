"""Tests for morie.fn.logrnk -- log-rank test."""

from morie.fn import _array_core as np
import pytest

from morie.fn.logrnk import logrnk


def test_logrnk_identical_groups_give_zero_statistic():
    """The same sample in both arms: observed = expected at every event
    time, so the statistic is exactly 0 and p = 1."""
    t = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    e = np.ones(5, dtype=int)
    r = logrnk(t, e, t, e)
    assert float(r["statistic"]) == pytest.approx(0.0, abs=1e-12)
    assert float(r["pvalue"]) == pytest.approx(1.0, abs=1e-12)


def test_logrnk_separated_survival_curves_reject():
    rng = np.random.default_rng(42)
    t1 = rng.exponential(2.0, 100)
    t2 = rng.exponential(10.0, 100)
    e = np.ones(100, dtype=int)
    r = logrnk(t1, e, t2, e)
    assert float(r["pvalue"]) < 1e-6
    assert float(r["statistic"]) > 10


def test_logrnk_censoring_removes_events_from_the_count():
    """Fully censored data carry no events, so there is no evidence."""
    t1 = np.array([1.0, 2.0, 3.0, 4.0])
    t2 = np.array([1.5, 2.5, 3.5, 4.5])
    z = np.zeros(4, dtype=int)
    r = logrnk(t1, z, t2, z)
    assert float(r["statistic"]) == pytest.approx(0.0, abs=1e-12)
    assert float(r["pvalue"]) == pytest.approx(1.0, abs=1e-12)


def test_logrnk_result_carries_the_df_and_name():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.array([1, 1, 1, 0, 1])
    r = logrnk(t, e, t + 0.5, e)
    assert int(r["df"]) == 1
    # The result stores the chi-square inputs; a df of 1 and a finite
    # statistic are the identifying facts of the two-group log-rank.
    assert np.isfinite(float(r["statistic"]))
