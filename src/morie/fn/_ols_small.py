"""Small dense OLS for the design-based spatial models (no numpy)."""

import math


def inverse(A):
    """Inverse of a symmetric positive-definite matrix by Gauss-Jordan with pivoting."""
    n = len(A)
    M = [list(map(float, row)) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12 * max(1.0, max(abs(v) for v in M[c][:n])):
            raise ValueError("design matrix is rank deficient")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def ols(X, y):
    """Coefficients, standard errors, sigma^2, residual df and residuals."""
    n, p = len(X), len(X[0])
    xtx = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    inv = inverse(xtx)
    beta = [sum(inv[a][b] * xty[b] for b in range(p)) for a in range(p)]
    res = [y[i] - sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    df = n - p
    if df < 1:
        raise ValueError("no residual degrees of freedom")
    s2 = sum(r * r for r in res) / df
    se = [math.sqrt(s2 * inv[a][a]) for a in range(p)]
    return beta, se, s2, df, res
