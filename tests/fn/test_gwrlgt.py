"""Tests for morie.fn.gwrlgt: the converged local-scoring fixed point."""

import math

from morie.fn import _array_core as np
from morie.fn.gwrlgt import gwrlgt

N = 20
P = [(float(i % 5), float(i // 5)) for i in range(N)]
X = [[(0.37 * i) % 1.3] for i in range(N)]
Y = [float((i * 7) % 3 == 0) for i in range(N)]


def test_fixed_point_and_loglik():
    r = gwrlgt(Y, X, P, 6.0, kernel="gaussian")
    mu = r["fitted"]
    eta = [math.log(m / (1 - m)) for m in mu]
    z = np.array([eta[j] + (Y[j] - mu[j]) / (mu[j] * (1 - mu[j])) for j in range(N)])
    Xa = np.column_stack([np.ones(N), np.array([x[0] for x in X])])
    for i in (0, 13):
        w = [
            math.exp(-(math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) ** 2) / 72.0) * mu[j] * (1 - mu[j])
            for j in range(N)
        ]
        Wd = np.diag(np.array(w))
        b = np.linalg.inv(Xa.T @ Wd @ Xa) @ (Xa.T @ Wd @ z)
        assert abs(float(b[1]) - r["betas"][i][1]) < 1e-6
    ll = sum(y * math.log(m) + (1 - y) * math.log(1 - m) for y, m in zip(Y, mu))
    assert abs(r["loglik"] - ll) < 1e-12
