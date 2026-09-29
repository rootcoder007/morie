# morie.fn -- function file (rootcoder007/morie)
"""Shared kernels of the gravity / spatial-interaction front ends (``igrav*``).

Log-linear OLS, Poisson and negative-binomial (NB2) maximum likelihood by
Fisher scoring, iterative proportional fitting, Wilson's doubly
constrained entropy model and Huff's retail model. R twin:
``R/GravityModels.R`` (helpers ``.grv_*``).
"""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._stats_core import _digamma, _trigamma

__all__: list = []


def vec(v):
    v = v.tolist() if hasattr(v, "tolist") else v
    return [float(t) for t in v]


def mat(A):
    A = A.tolist() if hasattr(A, "tolist") else A
    return [[float(v) for v in r] for r in A]


def _xtwx(X, w):
    p = len(X[0])
    return [[ssum(w[i] * X[i][a] * X[i][b] for i in range(len(X))) for b in range(p)] for a in range(p)]


def _mv(A, v):
    return [ssum(a * b for a, b in zip(r, v)) for r in A]


def design(mass_o, mass_d, dist):
    mo, md, d = vec(mass_o), vec(mass_d), vec(dist)
    if not (len(mo) == len(md) == len(d)):
        raise ValueError("mass_o, mass_d and dist must have the same length")
    if min(mo) <= 0 or min(md) <= 0 or min(d) <= 0:
        raise ValueError("masses and distances must be positive")
    return [[1.0, math.log(a), math.log(b), math.log(c)] for a, b, c in zip(mo, md, d)]


def ols(X, y):
    """OLS with the usual ``s^2 (X'X)^{-1}`` standard errors and R-squared."""
    n, p = len(X), len(X[0])
    A = inverse(_xtwx(X, [1.0] * n))
    A = [[float(v) for v in r] for r in A]
    b = _mv(A, [ssum(X[i][a] * y[i] for i in range(n)) for a in range(p)])
    e = [t - ssum(u * v for u, v in zip(r, b)) for r, t in zip(X, y)]
    sse = ssum(v * v for v in e)
    my = ssum(y) / n
    s2 = sse / (n - p)
    return {
        "coefficients": b,
        "se": [math.sqrt(s2 * A[k][k]) for k in range(p)],
        "sigma2": s2,
        "r2": 1.0 - sse / ssum((t - my) ** 2 for t in y),
        "residuals": e,
    }


def _nb_theta(y, mu, theta, tol=1e-13, max_iter=100):
    """Newton on the NB2 score for ``theta`` at fixed means (MASS::theta.ml)."""
    n = len(y)
    if theta is None:
        theta = n / ssum((a / m - 1.0) ** 2 for a, m in zip(y, mu))
    for _ in range(max_iter):
        s = ssum(
            _digamma(a + theta)
            - _digamma(theta)
            + math.log(theta)
            + 1.0
            - math.log(theta + m)
            - (a + theta) / (theta + m)
            for a, m in zip(y, mu)
        )
        i = -ssum(
            _trigamma(a + theta) - _trigamma(theta) + 1.0 / theta - 2.0 / (theta + m) + (a + theta) / (theta + m) ** 2
            for a, m in zip(y, mu)
        )
        if not i > 0.0 or theta > 1e10:
            raise ValueError("no overdispersion: theta diverges; fit the Poisson model (gravpp) instead")
        step = s / i
        while theta + step <= 0:  # keep theta positive: halve an overshooting Newton step
            step /= 2.0
        theta = theta + step
        if abs(step) < tol * max(1.0, theta):
            break
    return theta


def glm_fit(X, y, family="poisson", offset=None, tol=1e-13, max_iter=200):
    """Log-link Poisson or NB2 regression by Fisher scoring (IRLS), NB2 alternating with ``theta``.

    Starts at ``mu = (y + mean(y)) / 2``; stops when no coefficient moves by
    more than ``tol`` (relative). Returns coefficients, model-based and HC0
    standard errors, fitted means, log-likelihood and, for NB2, ``theta``.
    """
    n, p = len(X), len(X[0])
    off = [0.0] * n if offset is None else vec(offset)
    ybar = ssum(y) / n
    mu = [(a + ybar) / 2.0 for a in y]
    eta = [math.log(m) for m in mu]
    b = [0.0] * p
    theta = math.inf
    for it in range(max_iter):
        w = [m if family == "poisson" else m / (1.0 + m / theta) for m in mu]
        z = [e - o + (a - m) / m for e, o, a, m in zip(eta, off, y, mu)]
        A = [[float(v) for v in r] for r in inverse(_xtwx(X, w))]
        nb = _mv(A, [ssum(w[i] * X[i][a] * z[i] for i in range(n)) for a in range(p)])
        eta = [ssum(u * v for u, v in zip(r, nb)) + o for r, o in zip(X, off)]
        mu = [math.exp(v) for v in eta]
        change = max(abs(u - v) / max(1.0, abs(u)) for u, v in zip(nb, b))
        b = nb
        if family == "negbin":
            old = theta
            theta = _nb_theta(y, mu, None if it == 0 else theta)
            change = max(change, abs(theta - old) / theta if math.isfinite(old) else 1.0)
        if change < tol:
            break
    w = [m if family == "poisson" else m / (1.0 + m / theta) for m in mu]
    A = [[float(v) for v in r] for r in inverse(_xtwx(X, w))]
    # HC0 sandwich of the score: sum x x' (w (y - mu) / mu)^2
    sc = [(wi * (a - m) / m) ** 2 for wi, a, m in zip(w, y, mu)]
    M = _xtwx(X, sc)
    V = [[ssum(A[a][k] * M[k][m] * A[m][c] for k in range(p) for m in range(p)) for c in range(p)] for a in range(p)]
    if family == "poisson":
        ll = ssum(a * math.log(m) - m - math.lgamma(a + 1.0) for a, m in zip(y, mu))
    else:
        ll = ssum(
            math.lgamma(theta + a)
            - math.lgamma(theta)
            - math.lgamma(a + 1.0)
            + theta * math.log(theta)
            + (a * math.log(m) if a > 0 else 0.0)
            - (theta + a) * math.log(theta + m)
            for a, m in zip(y, mu)
        )
    out = {
        "coefficients": b,
        "se": [math.sqrt(A[k][k]) for k in range(p)],
        "se_robust": [math.sqrt(V[k][k]) for k in range(p)],
        "fitted": mu,
        "loglik": ll,
        "iterations": it + 1,
    }
    if family == "negbin":
        out["theta"] = theta
    return out


def ipf(M, rows, cols, tol=1e-12, max_iter=10000):
    """Iterative proportional fitting of a non-negative matrix to row and column totals (Deming-Stephan)."""
    T = mat(M)
    r, c = vec(rows), vec(cols)
    if abs(ssum(r) - ssum(c)) > 1e-9 * max(1.0, ssum(r)):
        raise ValueError("row and column totals must have the same sum")
    ni, nj = len(T), len(T[0])
    err = math.inf
    iters = 0
    for _ in range(max_iter):
        iters += 1
        for i in range(ni):
            s = ssum(T[i])
            if s > 0:
                f = r[i] / s
                T[i] = [v * f for v in T[i]]
        for j in range(nj):
            s = ssum(T[i][j] for i in range(ni))
            if s > 0:
                f = c[j] / s
                for i in range(ni):
                    T[i][j] *= f
        err = max(abs(ssum(T[i]) - r[i]) for i in range(ni))
        if err < tol * max(1.0, max(r)):
            break
    return T, err, iters


def wilson(orig, dest, C, beta, tol=1e-12, max_iter=10000):
    """Doubly constrained entropy-maximising model ``T_ij = A_i O_i B_j D_j exp(-beta c_ij)`` (Wilson 1967)."""
    og, dt, C = vec(orig), vec(dest), mat(C)
    ni, nj = len(og), len(dt)
    f = [[math.exp(-beta * C[i][j]) for j in range(nj)] for i in range(ni)]
    B = [1.0] * nj
    iters = 0
    for _ in range(max_iter):
        iters += 1
        A = [1.0 / ssum(B[j] * dt[j] * f[i][j] for j in range(nj)) for i in range(ni)]
        Bn = [1.0 / ssum(A[i] * og[i] * f[i][j] for i in range(ni)) for j in range(nj)]
        ch = max(abs(u - v) / abs(u) for u, v in zip(Bn, B))
        B = Bn
        if ch < tol:
            break
    A = [1.0 / ssum(B[j] * dt[j] * f[i][j] for j in range(nj)) for i in range(ni)]
    T = [[A[i] * og[i] * B[j] * dt[j] * f[i][j] for j in range(nj)] for i in range(ni)]
    return T, A, B, iters


def huff(mass, dist, beta):
    """Huff (1963) probabilities ``P_ij = (M_j / d_ij^beta) / sum_k (M_k / d_ik^beta)``."""
    M, D = vec(mass), mat(dist)
    P = []
    for row in D:
        u = [m / d**beta for m, d in zip(M, row)]
        s = ssum(u)
        P.append([v / s for v in u])
    return P
