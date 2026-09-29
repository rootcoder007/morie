"""Tests for morie.fn.gwrpois: the converged local-scoring fixed point."""

import math

from morie.fn import _array_core as np
from morie.fn.gwrpois import gwrpois

N = 20
P = [(float(i % 5), float(i // 5)) for i in range(N)]
X = [[(0.37 * i) % 1.3] for i in range(N)]
Y = [float((i * 7) % 5 + (i // 5)) for i in range(N)]


def test_fixed_point_and_deviance():
    r = gwrpois(Y, X, P, 4.0, kernel="gaussian")
    mu = r["fitted"]
    eta = [math.log(v) for v in mu]
    z = np.array([eta[j] + (Y[j] - mu[j]) / mu[j] for j in range(N)])
    Xa = np.column_stack([np.ones(N), np.array([x[0] for x in X])])
    for i in (0, 11):
        w = [math.exp(-(math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) ** 2) / 32.0) * mu[j] for j in range(N)]
        Wd = np.diag(np.array(w))
        b = np.linalg.inv(Xa.T @ Wd @ Xa) @ (Xa.T @ Wd @ z)
        assert abs(float(b[1]) - r["betas"][i][1]) < 1e-6
    dev = sum(2 * (y * math.log(y / m) - (y - m)) if y > 0 else 2 * m for y, m in zip(Y, mu))
    assert abs(r["deviance"] - dev) < 1e-9
    ll = sum(y * math.log(m) - m - math.lgamma(y + 1) for y, m in zip(Y, mu))
    assert abs(r["loglik"] - ll) < 1e-9
