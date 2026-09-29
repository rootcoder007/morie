"""Tests for morie.fn.cdyll: values recomputed from the definition."""

from morie.fn.cdyll import years_life_lost


def test_yll_is_deaths_times_remaining_life_expectancy():
    d, le = [2, 1, 3, 5], [30.5, 12.0, 4.0, 1.5]
    r = years_life_lost(d, le)
    assert abs(r.estimate - sum(a * b for a, b in zip(d, le))) < 1e-12
    assert r.extra["total_deaths"] == 11
