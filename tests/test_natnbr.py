"""Tests for morie.fn.natnbr (natural-neighbour interpolation)."""

import math

import pytest

from morie.fn import natnbr as N

P = [(0.3, 0.1), (2.1, 0.4), (1.7, 2.2), (0.2, 1.9), (1.1, 1.0), (2.9, 1.6), (0.9, 2.8), (2.4, 2.9)]


@pytest.mark.parametrize("method", ["sibson", "laplace"])
def test_linear_precision_and_partition_of_unity(method):
    for q in [(1.0, 1.2), (1.6, 0.9), (0.5, 1.5), (2.2, 2.3), (0.35, 0.3)]:
        w = N.natural_neighbour_weights(P, q, method=method).weights
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)
        assert sum(v * P[i][0] for i, v in w.items()) == pytest.approx(q[0], abs=1e-10)
        assert sum(v * P[i][1] for i, v in w.items()) == pytest.approx(q[1], abs=1e-10)
        assert all(v > 0 for v in w.values())


def test_interpolation_properties():
    z = [math.sin(x) + y for x, y in P]
    r = N.nn_interpolate(P, z, [P[4], (1.2, 1.3), (9.0, 9.0)])
    assert r.estimates[0] == z[4] and math.isnan(r.estimates[2])
    assert r.lower[1] <= r.estimates[1] <= r.upper[1] and r.variance[1] >= 0
    ext = N.nn_interpolate(P, z, [(9.0, 9.0)], outside="extrapolate")
    assert not math.isnan(ext.estimates[0])
    sq = N.natural_neighbour_weights([(0, 0), (1, 0), (1, 1), (0, 1)], (0.5, 0.5), method="laplace").weights
    assert [sq[i] for i in range(4)] == pytest.approx([0.25] * 4)


def test_gradient_cv_domain():
    lin = [2 * x - y + 1 for x, y in P]
    g = N.nn_gradient_interpolate(P, lin, [(1.3, 1.4)])
    assert g.estimates[0] == pytest.approx(2 * 1.3 - 1.4 + 1, abs=1e-10)
    assert g.gradients[4] == pytest.approx([2.0, -1.0], abs=1e-10)
    cv = N.nn_cross_validation(P, lin)
    assert cv.rmse == pytest.approx(0.0, abs=1e-10) and cv.n_predicted >= 1
    assert N.nn_in_domain(P, [(1.0, 1.0), (5.0, 5.0)]) == [True, False]
