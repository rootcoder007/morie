"""Tests for morie.fn.stsmod: Kalman log-likelihood equals the joint Gaussian density of y."""

import math

from morie.fn.stsmod import state_space_model

Y = [1.1, 0.9, 1.4, 1.2, 1.6, 1.3]


def _mvn_loglik(y, S):
    n = len(y)
    M = [list(r) + [v] for r, v in zip(S, y)]
    L = [[0.0] * n for _ in range(n)]
    for j in range(n):
        L[j][j] = math.sqrt(S[j][j] - sum(L[j][k] ** 2 for k in range(j)))
        for i in range(j + 1, n):
            L[i][j] = (S[i][j] - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]
    z = []
    for i in range(n):
        z.append((y[i] - sum(L[i][k] * z[k] for k in range(i))) / L[i][i])
    del M
    return -0.5 * (n * math.log(2 * math.pi) + 2 * sum(math.log(L[i][i]) for i in range(n)) + sum(v * v for v in z))


def test_local_level_likelihood_and_smoother():
    # local level: y_t = mu_t + e, mu_{t+1} = mu_t + n, mu_1 ~ N(0, P1): Cov(y_s, y_t) = P1 + Q min(s,t)-1 terms
    H, Q, P1 = 0.5, 0.2, 10.0
    n = len(Y)
    S = [[P1 + Q * min(s, t) + (H if s == t else 0.0) for t in range(n)] for s in range(n)]
    r = state_space_model(Y, 1, 1, H, Q, 1, a1=[0.0], P1=[[P1]])
    assert abs(r["loglik"] - _mvn_loglik(Y, S)) < 1e-10
    # smoothed mean of mu_1 = E(mu_1 | y) = Cov(mu_1, y) S^-1 y with Cov(mu_1, y_t) = P1
    import itertools  # noqa: F401

    Si = _inv(S)
    sm = sum(P1 * Si[t][u] * Y[u] for t in range(n) for u in range(n))
    assert abs(r["smoothed"][0][0] - sm) < 1e-10


def _inv(A):
    n = len(A)
    M = [list(r) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [r[n:] for r in M]
