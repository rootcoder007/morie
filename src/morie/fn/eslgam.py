# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generalized additive models by backfitting and local scoring (ESL Ch 9.1)."""

import math

from ._richresult import RichResult

__all__ = ["esl_gam"]


def _band_solve(A, b, bw):
    """Solve A x = b for a banded matrix (half-bandwidth bw) by Gaussian elimination without pivoting."""
    n = len(b)
    A = [r[:] for r in A]
    b = b[:]
    for k in range(n):
        for i in range(k + 1, min(n, k + bw + 1)):
            f = A[i][k] / A[k][k]
            if f != 0.0:
                for j in range(k, min(n, k + bw + 1)):
                    A[i][j] -= f * A[k][j]
                b[i] -= f * b[k]
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        x[k] = (b[k] - sum(A[k][j] * x[j] for j in range(k + 1, min(n, k + bw + 1)))) / A[k][k]
    return x


def _spline_smooth(x, r, lam, w=None, full=False):
    """Weighted cubic smoothing spline at the data: minimise sum w (r - f)^2 + lam int f''^2 (Reinsch form).

    Tied x are pooled with their weights; returns fitted values in the input order, or with
    ``full`` also the knots, the fitted values there and the second derivatives (zero at the ends).
    """
    n = len(x)
    w = [1.0] * n if w is None else w
    groups = {}
    for i, v in enumerate(x):
        groups.setdefault(v, []).append(i)
    u = sorted(groups)
    m = len(u)
    W = [sum(w[i] for i in groups[v]) for v in u]
    ybar = [sum(w[i] * r[i] for i in groups[v]) / W[k] for k, v in enumerate(u)]
    gam = []
    if m < 3 or lam == 0:
        fu = ybar
    else:
        h = [u[k + 1] - u[k] for k in range(m - 1)]
        # Q is m x (m-2) with three non-zeros per column, R is (m-2) x (m-2) tridiagonal
        Q = [[0.0] * (m - 2) for _ in range(m)]
        R = [[0.0] * (m - 2) for _ in range(m - 2)]
        for j in range(1, m - 1):
            c = j - 1
            Q[j - 1][c] = 1 / h[j - 1]
            Q[j][c] = -1 / h[j - 1] - 1 / h[j]
            Q[j + 1][c] = 1 / h[j]
            R[c][c] = (h[j - 1] + h[j]) / 3
            if c + 1 < m - 2:
                R[c][c + 1] = R[c + 1][c] = h[j] / 6
        A = [
            [
                R[a][b] + lam * sum(Q[i][a] * Q[i][b] / W[i] for i in range(max(0, a), min(m, a + 5)))
                for b in range(m - 2)
            ]
            for a in range(m - 2)
        ]
        rhs = [sum(Q[i][a] * ybar[i] for i in range(a, a + 3)) for a in range(m - 2)]
        gam = _band_solve(A, rhs, 2)
        fu = [
            ybar[i] - lam * sum(Q[i][c] * gam[c] for c in range(max(0, i - 2), min(m - 2, i + 1))) / W[i]
            for i in range(m)
        ]
    out = [0.0] * n
    for k, v in enumerate(u):
        for i in groups[v]:
            out[i] = fu[k]
    if full:
        return out, u, fu, [0.0] + gam + [0.0] if gam else [0.0] * m
    return out


def _linear_smooth(x, r, w=None):
    n = len(x)
    w = [1.0] * n if w is None else w
    sw = sum(w)
    xm = sum(wi * xi for wi, xi in zip(w, x)) / sw
    rm = sum(wi * ri for wi, ri in zip(w, r)) / sw
    sxx = sum(wi * (xi - xm) ** 2 for wi, xi in zip(w, x))
    b = sum(wi * (xi - xm) * (ri - rm) for wi, xi, ri in zip(w, x, r)) / sxx if sxx > 0 else 0.0
    return [rm + b * (xi - xm) for xi in x]


def _backfit(cols, z, w, smooth, alpha, F, max_iter, tol):
    n, p = len(z), len(cols)
    sw = sum(w)
    it, converged = 0, False
    for _ in range(max_iter):
        it += 1
        delta = 0.0
        alpha = sum(wi * (zi - sum(F[j][i] for j in range(p))) for i, (wi, zi) in enumerate(zip(w, z))) / sw
        for j in range(p):
            part = [z[i] - alpha - sum(F[k][i] for k in range(p) if k != j) for i in range(n)]
            new = smooth(j, part, w)
            mu = sum(wi * v for wi, v in zip(w, new)) / sw
            new = [v - mu for v in new]
            delta = max(delta, max(abs(a - b) for a, b in zip(new, F[j])))
            F[j] = new
        if delta < tol:
            converged = True
            break
    return alpha, F, it, converged


def esl_gam(X, y, g=None, lambdas=1.0, smoother="spline", max_iter=500, tol=1e-10, max_outer=100):
    r"""Additive model :math:`g(\mu) = \alpha + \sum_j f_j(X_j)` fitted by backfitting (ESL Alg. 9.1).

    Each :math:`f_j` is a cubic smoothing spline in :math:`X_j` with penalty
    :math:`\lambda_j` (ESL eq 9.7): the backfitting fixed point minimises
    :math:`\sum_i(y_i - \alpha - \sum_j f_j(x_{ij}))^2 + \sum_j\lambda_j\int f_j''^2`.
    Every :math:`f_j` is re-centred to mean zero, which identifies
    :math:`\alpha`. With ``g="logit"`` the local-scoring algorithm (ESL
    Alg. 9.2) repeats weighted backfitting on the working response
    :math:`z = \eta + (y - p)/(p(1-p))` with weights :math:`p(1-p)` (eqs 9.3,
    9.8), maximising the penalised log-likelihood. ``smoother="linear"``
    uses a straight line per coordinate (with the identity link this is
    exactly least squares).

    Parameters
    ----------
    X : array-like, shape (n, p)
        Predictors, WITHOUT an intercept column.
    y : array-like, shape (n,)
        Response (0/1 for ``g="logit"``).
    g : None, "identity" or "logit"
        Link function.
    lambdas : float or sequence of p floats
        Smoothing-spline penalties (ignored by the linear smoother).
    smoother : {"spline", "linear"}
    max_iter, tol, max_outer
        Backfitting and local-scoring controls.

    Returns
    -------
    result : dict
        Keys: estimate (intercept alpha), alpha, partial_fits (p lists,
        one per predictor), fitted (the mean, probabilities for logit),
        eta, iterations, converged, rss (identity) or loglik (logit), n,
        p, method.

    References
    ----------
    Hastie, T. & Tibshirani, R. (1990). Generalized Additive Models.
    Chapman & Hall; Hastie, Tibshirani and Friedman (2009), Ch 9.1.
    """
    if g not in (None, "identity", "logit"):
        raise ValueError(f"link '{g}' is not implemented; use 'identity' or 'logit'.")
    if smoother not in ("spline", "linear"):
        raise ValueError("smoother must be 'spline' or 'linear'.")
    rows = [[float(v) for v in r] for r in X]
    yy = [float(v) for v in y]
    n, p = len(rows), len(rows[0])
    if len(yy) != n:
        raise ValueError(f"X has {n} rows but y has {len(yy)} entries.")
    lam = [float(lambdas)] * p if isinstance(lambdas, (int, float)) else [float(v) for v in lambdas]
    if len(lam) != p or min(lam) < 0:
        raise ValueError("lambdas must be one non-negative value or one per predictor.")
    cols = [[r[j] for r in rows] for j in range(p)]

    def smooth(j, r, w):
        if smoother == "linear":
            return _linear_smooth(cols[j], r, w)
        return _spline_smooth(cols[j], r, lam[j], w)

    F = [[0.0] * n for _ in range(p)]
    if g in (None, "identity"):
        alpha, F, it, converged = _backfit(cols, yy, [1.0] * n, smooth, sum(yy) / n, F, max_iter, tol)
        eta = [alpha + sum(F[j][i] for j in range(p)) for i in range(n)]
        extra = {"rss": sum((a - b) ** 2 for a, b in zip(yy, eta))}
        fitted = eta
    else:
        if any(v not in (0.0, 1.0) for v in yy):
            raise ValueError("the logit link needs 0/1 responses.")
        ybar = sum(yy) / n
        alpha = math.log(ybar / (1 - ybar))
        it, converged = 0, False
        for _ in range(max_outer):
            it += 1
            eta = [alpha + sum(F[j][i] for j in range(p)) for i in range(n)]
            pr = [1 / (1 + math.exp(-e)) for e in eta]
            w = [q * (1 - q) for q in pr]
            z = [e + (yv - q) / wv for e, yv, q, wv in zip(eta, yy, pr, w)]
            old = [alpha] + [v for f in F for v in f]
            alpha, F, _, _ = _backfit(cols, z, w, smooth, alpha, [f[:] for f in F], max_iter, tol)
            new = [alpha] + [v for f in F for v in f]
            if max(abs(a - b) for a, b in zip(new, old)) < tol:
                converged = True
                break
        eta = [alpha + sum(F[j][i] for j in range(p)) for i in range(n)]
        fitted = [1 / (1 + math.exp(-e)) for e in eta]
        extra = {"loglik": sum(yv * e - math.log1p(math.exp(e)) for yv, e in zip(yy, eta))}
    payload = {
        "estimate": alpha,
        "alpha": alpha,
        "partial_fits": F,
        "fitted": fitted,
        "eta": eta,
        "iterations": it,
        "converged": converged,
        "n": n,
        "p": p,
        "method": f"{'local scoring (ESL Alg. 9.2)' if g == 'logit' else 'backfitting (ESL Alg. 9.1)'}, "
        f"{smoother} smoothers, components centred",
    }
    payload.update(extra)
    return RichResult(payload=payload)


def cheatsheet():
    return "eslgam: backfit cubic smoothing splines (Alg. 9.1); logit by local scoring (Alg. 9.2)"


# compact alias per ledger/NAMING.md
eslgam = esl_gam
