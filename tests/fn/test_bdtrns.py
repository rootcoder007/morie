"""Tests for bdtrns (Manski worst-case bounds).

Replaces the generated stub, which imported a ``bound_transport`` that
does not exist and asserted only that the result was a dict.
"""

from morie.fn.bdtrns import bdtrns


def test_bounds_are_the_closed_form():
    # observed mean 4, 60% observed, outcome in [0, 10]
    y = [2.0, 4.0, 6.0]
    res = bdtrns(y, 0.6, 0.0, 10.0)
    mean = 4.0
    assert abs(res["observed_mean"] - mean) < 1e-12
    assert abs(res["lower"] - (0.6 * mean + 0.4 * 0.0)) < 1e-12
    assert abs(res["upper"] - (0.6 * mean + 0.4 * 10.0)) < 1e-12
    assert abs(res["width"] - 0.4 * 10.0) < 1e-12


def test_full_observation_collapses_the_interval():
    res = bdtrns([1.0, 3.0], 1.0, 0.0, 10.0)
    assert abs(res["width"]) < 1e-12
    assert abs(res["lower"] - res["upper"]) < 1e-12
    assert abs(res["lower"] - 2.0) < 1e-12


def test_the_interval_widens_as_observation_falls():
    wide = bdtrns([5.0], 0.2, 0.0, 10.0)["width"]
    narrow = bdtrns([5.0], 0.9, 0.0, 10.0)["width"]
    assert wide > narrow
    assert abs(wide - 0.8 * 10.0) < 1e-12


def test_bounds_always_contain_the_observed_mean_when_it_is_inside():
    res = bdtrns([3.0, 5.0], 0.5, 0.0, 10.0)
    assert res["lower"] <= res["observed_mean"] <= res["upper"]


def test_validation():
    for call in (lambda: bdtrns([], 0.5, 0.0, 1.0),
                 lambda: bdtrns([1.0], 1.5, 0.0, 1.0),
                 lambda: bdtrns([1.0], -0.1, 0.0, 1.0),
                 lambda: bdtrns([1.0], 0.5, 5.0, 1.0),
                 lambda: bdtrns([99.0], 0.5, 0.0, 1.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
