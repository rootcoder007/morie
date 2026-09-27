"""Maximum-likelihood factor analysis core (the objective and statistics of stats::factanal)."""

import math

from . import _array_core as np


def eigh_desc(M):
    """Eigenvalues (descending) and eigenvectors (columns) of a symmetric matrix given as lists."""
    w, V = np.linalg.eigh(np.asarray(M))
    w = list(w)
    V = V.tolist()
    order = sorted(range(len(w)), key=lambda i: -w[i])
    return [w[i] for i in order], [[V[r][i] for i in order] for r in range(len(w))]


def corr_matrix(X):
    n = len(X)
    p = len(X[0])
    mu = [sum(r[j] for r in X) / n for j in range(p)]
    C = [[sum((r[a] - mu[a]) * (r[b] - mu[b]) for r in X) / (n - 1) for b in range(p)] for a in range(p)]
    return C


def to_corr(C):
    s = [math.sqrt(C[i][i]) for i in range(len(C))]
    return [[C[i][j] / (s[i] * s[j]) for j in range(len(C))] for i in range(len(C))]


def _solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[piv] = M[piv], M[k]
        if abs(M[k][k]) < 1e-300:
            raise ValueError("singular system")
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= f * M[k][j]
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        x[k] = (M[k][n] - sum(M[k][j] * x[j] for j in range(k + 1, n))) / M[k][k]
    return x


def _inv(A):
    n = len(A)
    cols = [_solve(A, [float(i == j) for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _state(S, psi, m):
    p = len(S)
    sq = [math.sqrt(v) for v in psi]
    Ss = [[S[i][j] / (sq[i] * sq[j]) for j in range(p)] for i in range(p)]
    th, U = eigh_desc(Ss)
    F = sum(t - math.log(t) for t in th[m:]) - (p - m)
    L = [[sq[i] * U[i][k] * math.sqrt(max(th[k] - 1, 0.0)) for k in range(m)] for i in range(p)]
    g = [(sum(L[j][k] ** 2 for k in range(m)) + psi[j] - S[j][j]) / psi[j] ** 2 for j in range(p)]
    return F, g, L


def mlfa_fit(S, m, lower=0.005, max_iter=1000, gtol=1e-10):
    """Minimise factanal's F(psi) = sum_{j>m} (theta_j - log theta_j) - (p - m) over psi in [lower, 1].

    theta are the eigenvalues of Psi^{-1/2} S Psi^{-1/2}; the analytic gradient is
    (Lambda Lambda' + Psi - S)_jj / psi_j^2. Projected BFGS on the free coordinates
    (inverse-Hessian update, reset whenever the active set changes) with a
    backtracking line search; stops when the projected gradient falls below ``gtol``
    or F stops decreasing. Bounds [lower, 1] as in factanal.
    """
    p = len(S)
    if not 1 <= m < p:
        raise ValueError("need 1 <= m < p")
    Sinv = _inv(S)
    psi = [min(1.0, max(lower, (1 - 0.5 * m / p) / Sinv[j][j])) for j in range(p)]
    F, g, L = _state(S, psi, m)

    def free_set(x, gr):
        return [j for j in range(p) if not ((x[j] <= lower and gr[j] > 0) or (x[j] >= 1 and gr[j] < 0))]

    free = free_set(psi, g)
    Hinv = [[float(a == b) for b in range(p)] for a in range(p)]
    stall = 0
    for _ in range(max_iter):
        if not free or max(abs(g[j]) for j in free) < gtol:
            break
        d = [0.0] * p
        for a in free:
            d[a] = -sum(Hinv[a][b] * g[b] for b in free)
        if sum(d[j] * g[j] for j in free) >= 0:
            Hinv = [[float(a == b) for b in range(p)] for a in range(p)]
            d = [(-g[j] if j in free else 0.0) for j in range(p)]
        t = 1.0
        while True:
            cand = [min(1.0, max(lower, psi[j] + t * d[j])) for j in range(p)]
            Fc, gc, Lc = _state(S, cand, m)
            if Fc <= F - 1e-4 * t * abs(sum(d[j] * g[j] for j in free)) or t < 1e-14:
                break
            t /= 2
        sv = [a - b for a, b in zip(cand, psi)]
        yv = [a - b for a, b in zip(gc, g)]
        drop = F - Fc
        psi, F, g, L = cand, Fc, gc, Lc
        newfree = free_set(psi, g)
        if newfree != free:
            Hinv = [[float(a == b) for b in range(p)] for a in range(p)]
            free = newfree
        else:
            sy = sum(sv[j] * yv[j] for j in free)
            if sy > 1e-300:
                Hy = [sum(Hinv[a][b] * yv[b] for b in free) for a in range(p)]
                yHy = sum(yv[a] * Hy[a] for a in free)
                for a in free:
                    for b in free:
                        Hinv[a][b] += (sy + yHy) * sv[a] * sv[b] / sy**2 - (Hy[a] * sv[b] + sv[a] * Hy[b]) / sy
        stall = stall + 1 if drop <= 1e-16 * max(1.0, abs(F)) else 0
        if stall >= 3:
            break
    return {"uniquenesses": psi, "loadings": L, "objective": F}


def fa_statistic(F, n, p, m):
    """factanal's Bartlett-corrected chi-square and its degrees of freedom."""
    dof = ((p - m) ** 2 - p - m) / 2
    stat = (n - 1 - (2 * p + 5) / 6 - 2 * m / 3) * F
    return stat, dof
