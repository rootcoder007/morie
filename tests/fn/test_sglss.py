"""Tests for morie.fn.sglss: mean squared error recomputed."""

from morie.fn.sglss import squared_error_loss


def test_mean_and_total_squared_error():
    p, o = [1.0, 2.0, 4.0, 0.5], [1.5, 2.0, 3.0, -0.5]
    r = squared_error_loss(p, o)
    sq = [(b - a) ** 2 for a, b in zip(p, o)]
    assert abs(r.statistic - sum(sq) / 4) < 1e-15
    assert abs(r.extra["total_loss"] - sum(sq)) < 1e-15
