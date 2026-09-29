"""Tests for morie.fn.ensav: recompute from the definition."""

from morie.fn.ensav import ensemble_average

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_average():
    S = [X, [v * 0.5 for v in X], [v - 1 for v in X]]
    a = ensemble_average(S).value
    for j in range(8):
        assert abs(a[j] - (X[j] + 0.5 * X[j] + X[j] - 1) / 3) < 1e-15
