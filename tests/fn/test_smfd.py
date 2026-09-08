"""smfd: penalised B-spline smoothing (P-splines).

The generated test imported `smooth_functional_data`, which does not
exist. Rewritten against penalized_spline and anchored on how the
roughness penalty is supposed to behave.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.smfd import penalized_spline

X = [i / 20.0 for i in range(21)]
Y = [x * x for x in X]


def test_fitted_values_match_the_input_length():
    r = penalized_spline(X, Y, nseg=8, lam=1.0)
    assert len(np.asarray(r["fitted"])) == len(X)
    assert all(np.isfinite(float(v)) for v in np.asarray(r["fitted"]))


def test_a_small_penalty_follows_a_smooth_curve_closely():
    """y = x^2 is representable in the cubic basis, so a light penalty
    should reproduce it almost exactly."""
    r = penalized_spline(X, Y, nseg=10, lam=1e-8)
    assert float(r["rss"]) == pytest.approx(0.0, abs=1e-6)


def test_more_penalty_means_fewer_effective_degrees_of_freedom():
    """The effective dimension is the trace of the hat matrix; increasing
    lambda must shrink it. This is the knob's entire purpose."""
    light = float(penalized_spline(X, Y, nseg=10, lam=1e-6)["effective_dimension"])
    heavy = float(penalized_spline(X, Y, nseg=10, lam=1e6)["effective_dimension"])
    assert heavy < light


def test_a_heavy_penalty_of_order_two_approaches_a_straight_line():
    """With order = 2 the null space of the penalty is linear functions, so
    lambda -> infinity drives the fit towards a straight line."""
    r = penalized_spline(X, Y, nseg=10, lam=1e10, order=2)
    f = [float(v) for v in np.asarray(r["fitted"])]
    second = [f[i + 1] - 2 * f[i] + f[i - 1] for i in range(1, len(f) - 1)]
    assert max(abs(s) for s in second) < 1e-3
