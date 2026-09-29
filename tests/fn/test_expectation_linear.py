"""Tests for morie.fn.expectation_linear: values recomputed from first principles."""

from morie.fn.expectation_linear import expectation_linear


def test_linearity_on_a_joint_pmf():
    # X, Y dependent: linearity holds regardless
    pts = [((0, 1), 0.2), ((1, 1), 0.3), ((1, 3), 0.1), ((2, 0), 0.4)]
    ex = sum(p * x for (x, _), p in pts)
    ey = sum(p * y for (_, y), p in pts)
    direct = sum(p * (2 * x - 3 * y + 5) for (x, y), p in pts)
    assert abs(expectation_linear(2, ex, -3, ey, 5)["expectation"] - direct) < 1e-14
