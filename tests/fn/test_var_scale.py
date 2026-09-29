"""Tests for morie.fn.var_scale: values recomputed from first principles."""

from morie.fn.var_scale import var_scale


def test_variance_scaling_on_a_pmf():
    v, p = [1.0, 2.0, 6.0], [0.2, 0.5, 0.3]
    var = lambda vals: sum(q * (a - sum(q2 * b for b, q2 in zip(vals, p))) ** 2 for a, q in zip(vals, p))  # noqa: E731
    assert abs(var_scale(-1.5, var(v))["var_aX"] - var([-1.5 * a for a in v])) < 1e-13
