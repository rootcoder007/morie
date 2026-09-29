"""Tests for morie.fn.carjac: 0.5 log|I - rho W| from the eigenvalues."""

import math

from morie.fn.carjac import carjac

W = [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]


def test_path_eigenvalues():
    # eigenvalues of the path graph P4 are 2 cos(k pi / 5), k = 1..4
    rho = 0.35
    ref = 0.5 * sum(math.log(1 - rho * 2 * math.cos(k * math.pi / 5)) for k in range(1, 5))
    assert abs(carjac(W, rho).statistic - ref) < 1e-14
