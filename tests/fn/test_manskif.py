"""Tests for manskif (Manski worst-case bounds).

Replaces the generated stub, which imported ``manski_bounds``.
"""

from morie.fn.manskif import manskif


def test_the_bounds_are_the_closed_form():
    y = [2.0, 4.0, 6.0, 8.0]
    observed = [1, 1, 0, 0]            # only the first two are seen
    res = manskif(y, observed, (0.0, 10.0))
    seen = [2.0, 4.0]
    p = 0.5
    mean = sum(seen) / len(seen)
    assert abs(res["lower"] - (p * mean + (1 - p) * 0.0)) < 1e-9
    assert abs(res["upper"] - (p * mean + (1 - p) * 10.0)) < 1e-9


def test_full_observation_pins_the_mean():
    y = [1.0, 3.0]
    res = manskif(y, [1, 1], (0.0, 10.0))
    assert abs(res["upper"] - res["lower"]) < 1e-12
    assert abs(res["lower"] - 2.0) < 1e-12


def test_the_width_is_the_missing_share_times_the_support():
    y = [2.0] * 10                     # inside the declared support
    res = manskif(y, [1] * 3 + [0] * 7, (0.0, 4.0))
    assert abs((res["upper"] - res["lower"]) - 0.7 * 4.0) < 1e-9


def test_an_outcome_outside_the_support_is_refused():
    # the support is the only assumption the method makes, so an
    # observation that violates it voids the bounds rather than being
    # quietly clipped
    try:
        manskif([5.0, 1.0], [1, 1], (0.0, 4.0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_no_assumption_ate_bounds_always_straddle_zero():
    # Manski's own point: with two arms and no assumptions the bounds
    # have width exactly equal to the support, so they can never sign an
    # effect. The module states that identity and it holds here.
    y = [1.0, 2.0, 8.0, 9.0]
    res = manskif(y, [1, 1, 1, 0], (0.0, 10.0), treatment=[0, 0, 1, 1])
    assert abs(res["ate_width"] - 10.0) < 1e-9
    assert res["ate_lower"] <= 0.0 <= res["ate_upper"]
    assert res["contains_zero"] is True
    assert res["identified"] is False
    assert res["y0_bounds"][0] <= res["y0_bounds"][1]
    assert res["y1_bounds"][0] <= res["y1_bounds"][1]
