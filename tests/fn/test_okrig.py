"""Tests for okrig.ordinary_kriging."""

from morie.fn import _array_core as np
import pytest

from morie.fn.okrig import ordinary_kriging


def test_okrig_is_exact_at_a_data_location():
    """With zero nugget, kriging interpolates: predicting AT a sample point
    returns the observed value with zero kriging variance (S&G 2005,
    Sec. 5.2.2)."""
    x = [1.0, 3.0, 2.0, 5.0]
    coords = [[0.0], [1.0], [2.0], [3.0]]
    r = ordinary_kriging(x, coords, [[1.0]], nugget=0.0, sill=1.0, range_=1.0)
    assert float(r["estimate"]) == pytest.approx(3.0, abs=1e-8)
    assert float(r["se"]) == pytest.approx(0.0, abs=1e-6)


def test_okrig_weights_sum_to_one_via_a_constant_field():
    """The unbiasedness constraint lambda'1 = 1 means a constant field is
    predicted exactly, anywhere."""
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    r = ordinary_kriging([7.0] * 4, coords, [[0.3, 0.6], [2.0, 2.0]])
    for est in r["estimate"]:
        assert float(est) == pytest.approx(7.0, abs=1e-8)


def test_okrig_symmetric_configuration_averages():
    """A target equidistant from two data points gets weight 1/2 each, so
    the prediction is their mean."""
    r = ordinary_kriging([2.0, 6.0], [[0.0], [2.0]], [[1.0]])
    assert float(r["estimate"]) == pytest.approx(4.0, abs=1e-8)


def test_okrig_variance_grows_with_distance():
    """sigma^2_ok(s0) = C(0) - lambda' sigma + m rises toward the sill as the
    target moves away from all data (eq. 5.16)."""
    x = [1.0, 2.0, 3.0]
    coords = [[0.0], [1.0], [2.0]]
    ses = [
        float(ordinary_kriging(x, coords, [[t]], sill=2.0, range_=1.0)["se"])
        for t in (1.0, 3.0, 6.0, 12.0)
    ]
    assert ses[0] < ses[1] < ses[2] < ses[3]
    # Far from all data the kriging variance approaches C(0) + the
    # Lagrange term; it must at least reach the sill.
    assert ses[-1] ** 2 >= 2.0 - 1e-6


def test_okrig_all_three_models_stay_exact():
    for model in ("exponential", "gaussian", "spherical"):
        r = ordinary_kriging([1.0, 4.0], [[0.0], [1.0]], [[0.0]], model=model)
        assert float(r["estimate"]) == pytest.approx(1.0, abs=1e-6), model


def test_okrig_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown model"):
        ordinary_kriging([1.0, 2.0], [[0.0], [1.0]], [[0.5]], model="cubic")
    with pytest.raises(ValueError, match="target dim"):
        ordinary_kriging([1.0, 2.0], [[0.0], [1.0]], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="coords rows"):
        ordinary_kriging([1.0, 2.0, 3.0], [[0.0], [1.0]], [[0.5]])
    with pytest.raises(ValueError, match="sill must be"):
        ordinary_kriging([1.0, 2.0], [[0.0], [1.0]], [[0.5]], nugget=2.0, sill=1.0)
