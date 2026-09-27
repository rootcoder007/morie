"""Penalised logistic regression on a basis expansion (ESL sec 5.6)."""

import math

from ._richresult import RichResult
from .nlsgn import _inverse

__all__ = ["esl_penalized_logistic"]


def esl_penalized_logistic(N, y, Omega, lambda_, max_iter=100, tol=1e-10):
    r"""Maximise :math:`\ell(\theta) - \tfrac{\lambda}{2}\theta^T\Omega\theta` for the logit :math:`f(x) = N(x)^T\theta`.

    ESL eqs 5.28-5.33: :math:`\log[\Pr(Y=1|x)/\Pr(Y=0|x)] = f(x)`, and the
    Newton step is the penalised weighted least squares
    :math:`\theta^{new} = (N^TWN + \lambda\Omega)^{-1}N^TWz` with
    :math:`z = N\theta^{old} + W^{-1}(y - p)`. At the solution
    :math:`N^T(y - p) = \lambda\Omega\theta`. The effective degrees of freedom
    are :math:`\mathrm{tr}\,N(N^TWN+\lambda\Omega)^{-1}N^TW`.

    Parameters
    ----------
    N : n x m nested sequence
        Basis matrix (include a constant column for an intercept; leave its
        row and column of Omega zero so it is not penalised).
    y : sequence of 0/1
    Omega : m x m nested sequence
        Penalty matrix, e.g. :math:`\int N''(t)N''(t)^Tdt` for a smoothing spline.
    lambda_ : float
        Penalty, >= 0.
    max_iter, tol
        Newton controls.

    Returns
    -------
    RichResult
        ``theta``, ``prob``, ``loglik``, ``penalized_loglik``, ``df``,
        ``iterations``, ``converged``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 5.6.
    """
    B = [[float(v) for v in r] for r in N]
    yy = [float(v) for v in y]
    Om = [[float(v) for v in r] for r in Omega]
    n, m = len(B), len(B[0])
    lam = float(lambda_)
    if len(yy) != n or len(Om) != m or any(len(r) != m for r in Om) or lam < 0:
        raise ValueError("need N (n x m), y of length n, Omega m x m and lambda >= 0")
    if any(v not in (0.0, 1.0) for v in yy):
        raise ValueError("y must be 0/1")
    th = [0.0] * m
    converged, it = False, 0
    for _ in range(max_iter):
        it += 1
        eta = [sum(B[i][j] * th[j] for j in range(m)) for i in range(n)]
        p = [1 / (1 + math.exp(-e)) for e in eta]
        w = [v * (1 - v) for v in p]
        z = [eta[i] + (yy[i] - p[i]) / w[i] for i in range(n)]
        A = [[sum(B[i][a] * w[i] * B[i][b] for i in range(n)) + lam * Om[a][b] for b in range(m)] for a in range(m)]
        rhs = [sum(B[i][a] * w[i] * z[i] for i in range(n)) for a in range(m)]
        inv = _inverse(A)
        new = [sum(inv[a][b] * rhs[b] for b in range(m)) for a in range(m)]
        step = max(abs(u - v) for u, v in zip(new, th))
        th = new
        if step < tol:
            converged = True
            break
    eta = [sum(B[i][j] * th[j] for j in range(m)) for i in range(n)]
    p = [1 / (1 + math.exp(-e)) for e in eta]
    w = [v * (1 - v) for v in p]
    A = [[sum(B[i][a] * w[i] * B[i][b] for i in range(n)) + lam * Om[a][b] for b in range(m)] for a in range(m)]
    inv = _inverse(A)
    btwb = [[sum(B[i][a] * w[i] * B[i][b] for i in range(n)) for b in range(m)] for a in range(m)]
    df = sum(inv[a][b] * btwb[b][a] for a in range(m) for b in range(m))
    ll = sum(yy[i] * eta[i] - math.log1p(math.exp(eta[i])) for i in range(n))
    pen = 0.5 * lam * sum(th[a] * Om[a][b] * th[b] for a in range(m) for b in range(m))
    return RichResult(
        title="Penalised logistic regression",
        summary_lines=[("loglik", ll), ("df", df)],
        payload={
            "theta": th,
            "prob": p,
            "loglik": ll,
            "penalized_loglik": ll - pen,
            "df": df,
            "iterations": it,
            "converged": converged,
        },
    )


def cheatsheet():
    return "eslplg: theta = (N'WN + lambda Omega)^-1 N'Wz iterated (penalised IRLS), ESL 5.33"
