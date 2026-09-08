# morie.fn -- function file (rootcoder007/morie)
"""Cook's distance for a linear model.

Source CONSULTED: Cook, R. D. (1977), "Detection of influential
observations in linear regression", *Technometrics* 19(1):15-18.  The
Technometrics article is paywalled; the two equivalent expressions used
here were checked against the standard published statement of them,

    D_i = sum_j (yhat_j - yhat_j(i))^2 / (p s^2)
        = e_i^2 / (p s^2) * h_ii / (1 - h_ii)^2

with p the rank of the design, s^2 the residual mean square, e_i the
i-th residual and h_ii the i-th leverage (the i-th diagonal entry of the
hat matrix H = X (X'X)^-1 X').  The computational form on the right is
the one implemented, so no observation is ever refitted.
"""

import math

from ._richresult import RichResult

__all__ = ["cooks_distance"]


def _solve(A, B):
    """Gauss-Jordan solve A Z = B for a matrix right-hand side."""
    n = len(A)
    m = len(B[0])
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            raise ValueError("X'X is singular; the design is rank deficient")
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            fac = M[r][c] / pv
            if fac == 0.0:
                continue
            for k in range(c, n + m):
                M[r][k] -= fac * M[c][k]
    return [[M[i][n + j] / M[i][i] for j in range(m)] for i in range(n)]


def cooks_distance(y, X):
    """Cook's distance for every observation of an OLS fit.

    Parameters
    ----------
    y : sequence, length n
        Response.
    X : array-like, shape (n, p)
        Design matrix.  An intercept is NOT added; include a constant
        column if you want one.

    Returns
    -------
    RichResult
        ``d`` (Cook's distance per observation), ``leverage``,
        ``residual``, ``std_residual``, ``beta``, ``sigma2``,
        ``max_d``, ``argmax_d``, ``threshold`` (the conventional 4/n
        screening cut), ``p``, ``n``.
    """
    yv = [float(v) for v in y]
    Xm = [[float(v) for v in row] for row in X]
    n = len(yv)
    if n == 0 or len(Xm) != n:
        raise ValueError("y and X must be non-empty and have the same length")
    p = len(Xm[0])
    if any(len(r) != p for r in Xm):
        raise ValueError("X must be rectangular")
    if n <= p:
        raise ValueError("need more observations than parameters")

    xtx = [[sum(Xm[i][a] * Xm[i][b] for i in range(n)) for b in range(p)]
           for a in range(p)]
    xty = [[sum(Xm[i][a] * yv[i] for i in range(n))] for a in range(p)]
    beta = [row[0] for row in _solve(xtx, xty)]
    # Columns of (X'X)^-1 via the same factorisation.
    eye = [[1.0 if a == b else 0.0 for b in range(p)] for a in range(p)]
    xtxi = _solve(xtx, eye)

    fitted = [sum(Xm[i][a] * beta[a] for a in range(p)) for i in range(n)]
    resid = [yv[i] - fitted[i] for i in range(n)]
    rss = sum(r * r for r in resid)
    sigma2 = rss / (n - p)
    lev = []
    for i in range(n):
        acc = 0.0
        for a in range(p):
            s = 0.0
            for b in range(p):
                s += xtxi[a][b] * Xm[i][b]
            acc += Xm[i][a] * s
        lev.append(acc)

    d = []
    stdres = []
    for i in range(n):
        h = lev[i]
        om = 1.0 - h
        if om <= 0.0:
            d.append(float("inf"))
            stdres.append(float("inf"))
            continue
        d.append(resid[i] * resid[i] / (p * sigma2) * h / (om * om)
                 if sigma2 > 0.0 else 0.0)
        stdres.append(resid[i] / math.sqrt(sigma2 * om) if sigma2 > 0.0 else 0.0)

    mx = max(d)
    return RichResult(payload={
        "d": [float(v) for v in d],
        "leverage": [float(v) for v in lev],
        "residual": [float(v) for v in resid],
        "std_residual": [float(v) for v in stdres],
        "beta": [float(v) for v in beta],
        "sigma2": float(sigma2), "rss": float(rss),
        "max_d": float(mx), "argmax_d": d.index(mx),
        "threshold": 4.0 / n, "p": p, "n": n,
        "method": "Cook (1977) distance, D_i = e_i^2 h_ii / (p s^2 (1 - h_ii)^2)"})


def cheatsheet():
    return "cookd: Cook (1977) distance for a linear model"


# compact alias per ledger/NAMING.md
cooksd = cooks_distance
