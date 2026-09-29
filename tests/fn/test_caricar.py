"""Tests for morie.fn.caricar: generalised-determinant ICAR log-density."""

import math

from morie.fn.caricar import caricar

W = [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
PHI = [0.4, -0.1, 0.3, -0.6]


def test_path_density():
    # Laplacian of P4 has eigenvalues 2 - 2 cos(k pi / 4), k = 0..3 (one zero)
    tau = 2.5
    pos = [2 - 2 * math.cos(k * math.pi / 4) for k in range(1, 4)]
    quad = sum((PHI[i] - PHI[i + 1]) ** 2 for i in range(3))
    ref = -1.5 * math.log(2 * math.pi) + 1.5 * math.log(tau) + 0.5 * sum(math.log(v) for v in pos) - 0.5 * tau * quad
    r = caricar(PHI, W, tau=tau)
    assert abs(r.statistic - ref) < 1e-12
    assert r.extra["rank"] == 3
