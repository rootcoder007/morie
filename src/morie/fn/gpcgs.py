# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Sparse variational GP classification.

Hensman, Matthews and Ghahramani (2015), "Scalable variational
Gaussian process classification", AISTATS 2015 (PMLR 38:351-360),
arXiv:1411.2005 -- title verified against the arXiv record.  The bound
is

    L = sum_i E_{q(f_i)}[ log p(y_i | f_i) ] - KL( q(u) || p(u) ),

with q(u) = N(m, S) over M inducing values, and the marginals

    mu_i    = a_i' m,
    var_i   = k_ii - a_i' (K_mm - S) a_i,   a_i = K_mm^{-1} k_mi.

The expectation is taken by Gauss-Hermite quadrature (nodes from the
Golub-Welsch tridiagonal, so both language arms generate the same
nodes rather than carrying a hard-coded table), and m and the
Cholesky factor of S are moved by fixed-step gradient ascent with
central-difference gradients -- deterministic, unlike the stochastic
natural gradients of the paper, so the two arms follow the same path.

At the initial point q(u) = p(u) the KL term is exactly zero; that,
and the monotone increase of the bound, are what the tests check.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gp_classification_svgp"]


def _gh(n):
    """Gauss-Hermite nodes and weights via the Golub-Welsch tridiagonal."""
    A = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        v = math.sqrt((i + 1) / 2.0)
        A[i][i + 1] = v
        A[i + 1][i] = v
    vals, vecs = core.jacobi(A)
    w = [math.sqrt(math.pi) * vecs[0][j] ** 2 for j in range(n)]
    return vals, w


def _k(A, B, ell, var):
    out = []
    for i in range(len(A)):
        row = []
        for j in range(len(B)):
            s = 0.0
            for c in range(len(A[i])):
                d = A[i][c] - B[j][c]
                s += d * d
            row.append(var * math.exp(-0.5 * s / (ell * ell)))
        out.append(row)
    return out


def _tri(v, M):
    L = [[0.0] * M for _ in range(M)]
    t = 0
    for i in range(M):
        for j in range(i + 1):
            L[i][j] = v[t]
            t += 1
    return L


def gp_classification_svgp(X, y, X_test=None, M=3, lengthscale=1.0, variance=1.0,
                           iters=40, step=0.05, nodes=11, jitter=1e-8):
    """Variational sparse GP classifier with a probit likelihood."""
    A = core.mat(X)
    n = len(A)
    if n == 0:
        raise ValueError("gp_classification_svgp: X is empty")
    yv = [int(v) for v in core.vec(y)]
    if len(yv) != n:
        raise ValueError("gp_classification_svgp: X and y have different lengths")
    for v in yv:
        if v not in (0, 1):
            raise ValueError("gp_classification_svgp: labels must be 0 or 1")
    m = int(M)
    if m < 1 or m > n:
        raise ValueError("gp_classification_svgp: M must lie between 1 and n")
    ell = float(lengthscale)
    var = float(variance)
    if ell <= 0 or var <= 0:
        raise ValueError("gp_classification_svgp: lengthscale and variance must be positive")
    idx = [int(round(i * (n - 1) / max(m - 1, 1))) for i in range(m)] if m > 1 else [0]
    Z = [A[i] for i in idx]
    Kmm = _k(Z, Z, ell, var)
    for i in range(m):
        Kmm[i][i] += float(jitter)
    Knm = _k(A, Z, ell, var)
    Ai = [core.cholsolve(Kmm, Knm[i]) for i in range(n)]
    Lp = core.chol(Kmm)
    xs, ws = _gh(int(nodes))

    def bound(theta):
        mu_u = theta[:m]
        L = _tri(theta[m:], m)
        S = core.matmul(L, [[L[j][i] for j in range(m)] for i in range(m)])
        tot = 0.0
        for i in range(n):
            mi = sum(Ai[i][j] * mu_u[j] for j in range(m))
            kii = var
            q = 0.0
            for a in range(m):
                for b in range(m):
                    q += Ai[i][a] * (Kmm[a][b] - S[a][b]) * Ai[i][b]
            vi = kii - q
            if vi < 1e-12:
                vi = 1e-12
            sgn = 1.0 if yv[i] == 1 else -1.0
            e = 0.0
            for t in range(len(xs)):
                z = sgn * (mi + math.sqrt(2.0 * vi) * xs[t])
                e += ws[t] * math.log(max(core.pnorm(z), 1e-300))
            tot += e / math.sqrt(math.pi)
        # KL(N(m, S) || N(0, Kmm)) = 0.5[tr(Kmm^-1 S) + m'Kmm^-1 m - M + log|Kmm|/|S|]
        tr = 0.0
        for j in range(m):
            col = core.cholsolve(Kmm, [S[i][j] for i in range(m)])
            tr += col[j]
        Kim = core.cholsolve(Kmm, mu_u)
        quad = sum(mu_u[j] * Kim[j] for j in range(m))
        logdetK = 2.0 * sum(math.log(Lp[i][i]) for i in range(m))
        dS = 0.0
        for i in range(m):
            dS += math.log(abs(L[i][i]) + 1e-300)
        kl = 0.5 * (tr + quad - m + logdetK - 2.0 * dS)
        return tot - kl, kl

    theta = [0.0] * m + [Lp[i][j] for i in range(m) for j in range(i + 1)]
    path = []
    h = 1e-5
    for _ in range(int(iters)):
        b0, _kl = bound(theta)
        path.append(b0)
        g = []
        for t in range(len(theta)):
            tp = list(theta)
            tm = list(theta)
            tp[t] += h
            tm[t] -= h
            g.append((bound(tp)[0] - bound(tm)[0]) / (2.0 * h))
        theta = [theta[t] + float(step) * g[t] for t in range(len(theta))]
    elbo, kl = bound(theta)
    path.append(elbo)
    mu_u = theta[:m]
    L = _tri(theta[m:], m)
    S = core.matmul(L, [[L[j][i] for j in range(m)] for i in range(m)])
    Xs = A if X_test is None else core.mat(X_test)
    Ksm = _k(Xs, Z, ell, var)
    p = []
    mus = []
    for j in range(len(Xs)):
        a = core.cholsolve(Kmm, Ksm[j])
        mj = sum(a[t] * mu_u[t] for t in range(m))
        q = 0.0
        for x in range(m):
            for b in range(m):
                q += a[x] * (Kmm[x][b] - S[x][b]) * a[b]
        vj = max(var - q, 1e-12)
        mus.append(mj)
        p.append(core.pnorm(mj / math.sqrt(1.0 + vj)))
    return RichResult(
        title="Sparse variational GP classification",
        summary_lines=[("n", n), ("inducing", m), ("elbo", elbo)],
        payload={
            "estimate": p[0],
            "p": p,
            "predicted": [1 if v >= 0.5 else 0 for v in p],
            "latent_mean": mus,
            "elbo": elbo,
            "kl": kl,
            "elbo_path": path,
            "n": n,
            "method": "variational bound of Hensman, Matthews & Ghahramani (2015) with Gauss-Hermite quadrature",
        },
    )


def cheatsheet():
    return "gpcgs: sparse variational GP classification"
