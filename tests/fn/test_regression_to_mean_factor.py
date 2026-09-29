"""Tests for morie.fn.regression_to_mean_factor: values recomputed from first principles."""

from morie.fn.regression_to_mean_factor import regression_to_mean_factor


def test_r_squared_times_first_score():
    r = regression_to_mean_factor(0.8, 10.0)
    assert abs(r["factor"] - 0.64) < 1e-15
    assert abs(r["yavg"] - 6.4) < 1e-14
