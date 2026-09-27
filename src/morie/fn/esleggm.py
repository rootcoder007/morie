"""Gaussian graphical model with known structure by the modified regression algorithm (ESL Alg. 17.1)."""

from ._richresult import RichResult
from .nlsgn import _inverse

__all__ = ["esl_ggm_fit"]


def _solve(A, b):
    Ai = _inverse(A)
    return [sum(Ai[i][j] * b[j] for j in range(len(b))) for i in range(len(b))]


def esl_ggm_fit(S, adjacency, max_iter=1000, tol=1e-12):
    r"""Maximum-likelihood :math:`\Sigma` and :math:`\Theta = \Sigma^{-1}` with :math:`\Theta_{jk} = 0` for missing edges.

    ESL Alg. 17.1 maximises :math:`\log\det\Theta - \mathrm{trace}(S\Theta)`
    (eq 17.11) subject to the zeros: starting from W = S, each column j
    solves the reduced system :math:`W_{11}^*\beta^* = s_{12}^*` over its
    neighbours only (eq 17.19), sets :math:`w_{12} = W_{11}\hat\beta`, and at the
    end :math:`\hat\theta_{22} = 1/(s_{22} - w_{12}^T\hat\beta)`,
    :math:`\hat\theta_{12} = -\hat\beta\hat\theta_{22}`. The fitted
    :math:`\hat\Sigma` equals S on the diagonal and on the edges (the
    positive-definite completion of S).

    Parameters
    ----------
    S : p x p nested sequence
        Empirical covariance matrix.
    adjacency : p x p nested sequence of bools/0-1
        True where an edge is present (the diagonal is ignored).
    max_iter, tol
        Convergence controls (largest change in W).

    Returns
    -------
    RichResult
        ``Sigma``, ``Theta``, ``iterations``, ``converged``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 17.3.1.
    """
    Sm = [[float(v) for v in r] for r in S]
    p = len(Sm)
    A = [[bool(adjacency[i][j]) and i != j for j in range(p)] for i in range(p)]
    if any(len(r) != p for r in Sm) or any(A[i][j] != A[j][i] for i in range(p) for j in range(p)):
        raise ValueError("S must be square and adjacency symmetric")
    W = [r[:] for r in Sm]
    betas = [[0.0] * p for _ in range(p)]
    it, converged = 0, False
    for _ in range(max_iter):
        it += 1
        delta = 0.0
        for j in range(p):
            nb = [k for k in range(p) if A[j][k]]
            beta = [0.0] * p
            if nb:
                bs = _solve([[W[a][b] for b in nb] for a in nb], [Sm[a][j] for a in nb])
                for k, v in zip(nb, bs):
                    beta[k] = v
            betas[j] = beta
            for i in range(p):
                if i != j:
                    new = sum(W[i][k] * beta[k] for k in range(p) if k != j)
                    delta = max(delta, abs(new - W[i][j]))
                    W[i][j] = W[j][i] = new
        if delta < tol:
            converged = True
            break
    Th = [[0.0] * p for _ in range(p)]
    for j in range(p):
        b = betas[j]
        t22 = 1 / (Sm[j][j] - sum(W[j][k] * b[k] for k in range(p) if k != j))
        Th[j][j] = t22
        for k in range(p):
            if k != j:
                Th[k][j] = -b[k] * t22
    Th = [[0.5 * (Th[i][j] + Th[j][i]) for j in range(p)] for i in range(p)]
    return RichResult(
        title="Gaussian graphical model (known structure)",
        summary_lines=[("iterations", it)],
        payload={"Sigma": W, "Theta": Th, "iterations": it, "converged": converged},
    )


def cheatsheet():
    return "esleggm: ESL Alg. 17.1, regress each node on its neighbours using the current W"
