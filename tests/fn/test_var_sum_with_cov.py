"""Tests for morie.fn.var_sum_with_cov: values recomputed from first principles."""

from morie.fn.var_sum_with_cov import var_sum_with_cov


def test_against_a_joint_pmf():
    pts = [((0, 1), 0.2), ((1, 1), 0.3), ((1, 3), 0.1), ((2, 0), 0.4)]
    E = lambda f: sum(p * f(x, y) for (x, y), p in pts)  # noqa: E731
    mx, my = E(lambda x, y: x), E(lambda x, y: y)
    vx = E(lambda x, y: (x - mx) ** 2)
    vy = E(lambda x, y: (y - my) ** 2)
    c = E(lambda x, y: (x - mx) * (y - my))
    direct = E(lambda x, y: (x + y - mx - my) ** 2)
    assert abs(var_sum_with_cov(vx, vy, c)["var_sum"] - direct) < 1e-14
