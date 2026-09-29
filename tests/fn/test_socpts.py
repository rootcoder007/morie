"""Tests for morie.fn.socpts: known optima of cone programs and complementary feasibility."""

import math

from morie.fn.socpts import second_order_cone

I2 = [[1.0, 0.0], [0.0, 1.0]]


def test_disc_minimum_is_on_the_boundary_opposite_the_gradient():
    for c in ([1.0, 0.0], [3.0, -4.0]):
        r = second_order_cone(c, [I2], [[0.0, 0.0]], [([0.0, 0.0], 1.0)])
        nc = math.hypot(*c)
        want = [-c[0] / nc, -c[1] / nc]
        assert max(abs(a - b) for a, b in zip(r["x"], want)) < 1e-8
        assert abs(r["objective"] + nc) < 1e-8


def test_infeasible_start_and_linear_constraints():
    r = second_order_cone([1.0, 1.0], [I2], [[-5.0, -5.0]], [([0.0, 0.0], 1.0)])
    want = 5.0 - 1.0 / math.sqrt(2.0)
    assert max(abs(v - want) for v in r["x"]) < 1e-8
    Z, z = [[0.0, 0.0]], [0.0]
    box = second_order_cone(
        [-1.0, -2.0],
        [Z, Z, Z, Z],
        [z, z, z, z],
        [([-1.0, 0.0], 1.0), ([0.0, -1.0], 1.0), ([1.0, 0.0], 0.0), ([0.0, 1.0], 0.0)],
    )
    assert max(abs(a - b) for a, b in zip(box["x"], [1.0, 1.0])) < 1e-8
    assert min(box["slack"]) > -1e-9
