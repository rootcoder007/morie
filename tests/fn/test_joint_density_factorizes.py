"""Tests for morie.fn.joint_density_factorizes: values recomputed from first principles."""

from morie.fn.joint_density_factorizes import joint_density_factorizes


def test_trapezoid_mass_of_the_outer_product():
    gx = [0.0, 0.5, 1.0, 2.0]
    dx = [0.2, 0.6, 0.5, 0.1]
    gy = [0.0, 1.0, 3.0]
    dy = [0.3, 0.4, 0.1]
    tx = sum((gx[i + 1] - gx[i]) * (dx[i] + dx[i + 1]) / 2 for i in range(3))
    ty = sum((gy[i + 1] - gy[i]) * (dy[i] + dy[i + 1]) / 2 for i in range(2))
    r = joint_density_factorizes(gx, dx, gy, dy)
    assert abs(r["total_mass"] - tx * ty) < 1e-14
    assert r["shape"] == [4, 3]
