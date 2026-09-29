"""Tests for morie.fn.carbym: REML of the Gaussian BYM model recomputed from the full covariance."""

import math

from morie.fn import _array_core as np
from morie.fn.carbym import carbym

N = 16
W = [[1.0 if abs(i // 4 - j // 4) + abs(i % 4 - j % 4) == 1 else 0.0 for j in range(N)] for i in range(N)]
Y = [0.5 * (i // 4) + 0.4 * (i % 4) + 0.8 * math.sin(2.3 * i) for i in range(N)]


def _reml(su, sv):
    """Full-matrix REML log-likelihood of y = mu 1 + u + v, Cov = su Q^+ + sv I."""
    Q = np.diag(np.array([sum(r) for r in W])) - np.array(W)
    J = np.ones((N, N)) / N
    Qp = np.linalg.inv(Q + J) - J  # Moore-Penrose inverse of a connected Laplacian
    S = su * Qp + sv * np.eye(N)
    Si = np.linalg.inv(S)
    one = np.ones(N)
    y = np.array(Y)
    a = float(one @ (Si @ one))
    mu = float(one @ (Si @ y)) / a
    r = y - mu
    ld = math.log(float(np.linalg.det(S)))
    return -0.5 * (ld + math.log(a) - math.log(N) + float(r @ (Si @ r)) + (N - 1) * math.log(2 * math.pi))


def test_reml_optimum():
    r = carbym(Y, W)
    su, sv = r["sigma2_spatial"], r["sigma2_unstructured"]
    assert 0.0 < r["phi"] < 1.0
    assert abs(_reml(su, sv) - r["reml_loglik"]) < 1e-9
    for du, dv in ((1e-3, 0.0), (-1e-3, 0.0), (0.0, 1e-3), (0.0, -1e-3), (1e-3, -1e-3)):
        assert _reml(su + du, sv + dv) < r["reml_loglik"]
