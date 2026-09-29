"""Tests for morie.fn.gaussian_sum_density: recompute Morin (2016) from the formula."""

import math

from morie.fn.gaussian_sum_density import gaussian_sum_density


def test_convolution():
    z, sx, sy = 1.0, 3.0, 4.0
    s2 = sx * sx + sy * sy
    ref = math.exp(-z * z / (2 * s2)) / math.sqrt(2 * math.pi * s2)
    r = gaussian_sum_density(z, sx, sy)
    assert abs(r["density"] - ref) < 1e-15
    assert r["sigma_sum"] == 5.0
    # numerical convolution of the two densities
    h = 0.01
    conv = h * math.fsum(
        math.exp(-((t * h) ** 2) / (2 * sx * sx))
        / math.sqrt(2 * math.pi * sx * sx)
        * math.exp(-((z - t * h) ** 2) / (2 * sy * sy))
        / math.sqrt(2 * math.pi * sy * sy)
        for t in range(-4000, 4001)
    )
    assert abs(conv - ref) < 1e-10
