"""Ising model for binary data: exact maximum likelihood (ESL sec 17.4)."""

import math

from ._richresult import RichResult
from .nlsgn import _inverse

__all__ = ["esl_ising_fit"]


def esl_ising_fit(X, edges, max_iter=100, tol=1e-12):
    r"""Fit :math:`p(x,\Theta) = \exp[\sum_j\theta_{j0}x_j + \sum_{(j,k)\in E}\theta_{jk}x_jx_k - \Phi(\Theta)]` exactly.

    ESL eqs 17.28-17.35 with the constant node :math:`X_0 \equiv 1` (Ex.
    17.10) giving the main effects. The log-likelihood is concave and its
    gradient is :math:`\hat E(X_jX_k) - E_\Theta(X_jX_k)` (eq 17.34), so Newton's
    method with the exact moments from enumerating all :math:`2^p` states
    reaches the MLE; it is the Poisson log-linear fit of the :math:`2^p` table
    (eq 17.45). Node conditionals are logistic (eq 17.30):
    :math:`\Pr(X_j=1|X_{-j}) = \sigma(\theta_{j0} + \sum_k\theta_{jk}x_k)`.

    Parameters
    ----------
    X : N x p nested sequence of 0/1
        p should be small (enumeration of 2^p states).
    edges : sequence of (j, k) pairs, 0-based
    max_iter, tol
        Newton controls.

    Returns
    -------
    RichResult
        ``main`` (theta_j0), ``edges`` (list of ((j, k), theta_jk)),
        ``loglik``, ``Phi``, ``iterations``, ``converged``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 17.4.1.
    """
    rows = [[int(v) for v in r] for r in X]
    n, p = len(rows), len(rows[0])
    E = [tuple(sorted((int(a), int(b)))) for a, b in edges]
    if any(v not in (0, 1) for r in rows for v in r) or any(a == b or not 0 <= a < p or not 0 <= b < p for a, b in E):
        raise ValueError("X must be 0/1 and edges must join distinct nodes")
    if p > 16:
        raise ValueError("exact enumeration is limited to p <= 16")

    def stats(x):
        return [x[j] for j in range(p)] + [x[a] * x[b] for a, b in E]

    states = [[(s >> j) & 1 for j in range(p)] for s in range(2**p)]
    T = [stats(x) for x in states]
    d = p + len(E)
    tbar = [sum(stats(r)[k] for r in rows) / n for k in range(d)]
    th = [0.0] * d

    def moments(t):
        e = [sum(a * b for a, b in zip(t, s)) for s in T]
        m = max(e)
        w = [math.exp(v - m) for v in e]
        z = sum(w)
        pr = [v / z for v in w]
        mean = [sum(pr[i] * T[i][k] for i in range(len(T))) for k in range(d)]
        cov = [
            [sum(pr[i] * T[i][a] * T[i][b] for i in range(len(T))) - mean[a] * mean[b] for b in range(d)]
            for a in range(d)
        ]
        return mean, cov, m + math.log(z)

    it, converged = 0, False
    for _ in range(max_iter):
        it += 1
        mean, cov, phi = moments(th)
        g = [a - b for a, b in zip(tbar, mean)]
        inv = _inverse(cov)
        step = [sum(inv[a][b] * g[b] for b in range(d)) for a in range(d)]
        th = [a + b for a, b in zip(th, step)]
        if max(abs(s) for s in step) < tol:
            converged = True
            break
    mean, cov, phi = moments(th)
    ll = n * (sum(a * b for a, b in zip(th, tbar)) - phi)
    return RichResult(
        title="Ising model (exact MLE)",
        summary_lines=[("loglik", ll)],
        payload={
            "main": th[:p],
            "edges": [(e, th[p + k]) for k, e in enumerate(E)],
            "loglik": ll,
            "Phi": phi,
            "iterations": it,
            "converged": converged,
        },
    )


def cheatsheet():
    return "eslisg: Newton on the Ising log-likelihood with exact 2^p moments; = Poisson log-linear fit"
