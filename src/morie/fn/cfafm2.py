# morie.fn -- function file (rootcoder007/morie)
"""Confirmatory factor analysis, several factors, cross-loadings allowed."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["cfa_multifactor"]

_MAXIT = 5000
_TOL = 1e-13


def _cov_or_S(X, p=None):
    """Return an item covariance matrix from data or pass one through."""
    M = core.mat(X)
    if not M:
        raise ValueError("empty input: X has no rows")
    q = len(M[0])
    if len(M) == q and q > 1 and all(
            abs(M[i][j] - M[j][i]) < 1e-12 for i in range(q) for j in range(q)):
        return [[float(v) for v in r] for r in M]
    n = len(M)
    if n < 2:
        raise ValueError("need at least two observations to form a covariance")
    mu = [sum(M[i][j] for i in range(n)) / n for j in range(q)]
    return [[sum((M[i][a] - mu[a]) * (M[i][b] - mu[b]) for i in range(n)) / (n - 1)
             for b in range(q)] for a in range(q)]


def _inv(A):
    """Inverse of a symmetric positive-definite matrix, column by column."""
    m = len(A)
    cols = [core.cholsolve(A, [1.0 if j == k else 0.0 for j in range(m)])
            for k in range(m)]
    return [[cols[b][a] for b in range(m)] for a in range(m)]


def _logdet(A):
    L = core.chol(A)
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(A)))


def _fa_em(S, mask):
    """Masked EM factor analysis (Rubin & Thayer 1982 E- and M-steps).

    ``mask[i][j]`` is 1 where item i is allowed to load on factor j.
    Factors are standardised and orthogonal, so the model covariance is
    Lambda Lambda' + Psi.  Iteration is deterministic: the same start
    and the same stopping test in both language arms.
    """
    p = len(S)
    k = len(mask[0])
    vals, vecs = core.jacobi(S)
    lam = [[0.0] * k for _ in range(p)]
    for j in range(k):
        idx = p - 1 - j
        sv = math.sqrt(max(vals[idx], 0.0))
        for i in range(p):
            lam[i][j] = sv * vecs[i][idx] * mask[i][j]
    psi = []
    for i in range(p):
        v = S[i][i] - sum(lam[i][j] ** 2 for j in range(k))
        psi.append(v if v > 1e-6 else 1e-6)
    it = 0
    for it in range(1, _MAXIT + 1):
        Sig = [[sum(lam[a][j] * lam[b][j] for j in range(k)) + (psi[a] if a == b else 0.0)
                for b in range(p)] for a in range(p)]
        Si = _inv(Sig)
        # beta = Lambda' Sigma^-1   (k x p)
        beta = [[sum(lam[a][j] * Si[a][b] for a in range(p)) for b in range(p)]
                for j in range(k)]
        # Czz = I - beta Lambda + beta S beta'
        bS = [[sum(beta[j][a] * S[a][b] for a in range(p)) for b in range(p)]
              for j in range(k)]
        Czz = [[(1.0 if u == v else 0.0)
                - sum(beta[u][a] * lam[a][v] for a in range(p))
                + sum(bS[u][a] * beta[v][a] for a in range(p))
                for v in range(k)] for u in range(k)]
        Cxz = [[bS[j][i] for j in range(k)] for i in range(p)]
        delta = 0.0
        for i in range(p):
            act = [j for j in range(k) if mask[i][j]]
            new = [0.0] * k
            if act:
                A = [[Czz[u][v] for v in act] for u in act]
                b = [Cxz[i][u] for u in act]
                sol = core.ridgesolve(A, b, 1e-12)
                for t, j in enumerate(act):
                    new[j] = sol[t]
            q = S[i][i] - 2.0 * sum(new[j] * Cxz[i][j] for j in range(k)) + sum(
                new[u] * Czz[u][v] * new[v] for u in range(k) for v in range(k))
            q = q if q > 1e-8 else 1e-8
            delta = max(delta, abs(q - psi[i]),
                        max(abs(new[j] - lam[i][j]) for j in range(k)))
            lam[i] = new
            psi[i] = q
        if delta < _TOL:
            break
    Sig = [[sum(lam[a][j] * lam[b][j] for j in range(k)) + (psi[a] if a == b else 0.0)
            for b in range(p)] for a in range(p)]
    Si = _inv(Sig)
    fml = _logdet(Sig) - _logdet(S) + sum(
        S[a][b] * Si[b][a] for a in range(p) for b in range(p)) - p
    resid = max(abs(S[a][b] - Sig[a][b]) for a in range(p) for b in range(p))
    return lam, psi, fml, resid, it


def cfa_multifactor(X, factor_pattern):
    """
    CFA multi-factor with cross-loadings allowed

    Formula: X = Lambda F + eps; F ~ N(0, Phi)

    Fitted by the Rubin-Thayer EM algorithm with the loading pattern
    imposed: an entry of ``factor_pattern`` that is zero forces the
    corresponding loading to stay zero, so cross-loadings are estimated
    only where the confirmatory model allows them.  Factors are
    standardised and orthogonal (Phi = I), which is the identification
    the EM E-step assumes.

    Parameters
    ----------
    X : array-like
        n x p data matrix, or a p x p item covariance matrix.
    factor_pattern : array-like
        p x k matrix of 0/1 flags: 1 where item i may load on factor j.

    Returns
    -------
    result : dict
        Keys: estimate (variance explained), loadings, uniquenesses,
        fml, max_resid, communality, n_iter, p, k.

    References
    ----------
    Joreskog (1969), Psychometrika 34(2):183-202.
    Rubin & Thayer (1982), Psychometrika 47(1):69-76.
    """
    P = core.mat(factor_pattern)
    if not P:
        raise ValueError("empty input: factor_pattern is empty")
    S = _cov_or_S(X)
    p = len(S)
    if len(P) != p:
        raise ValueError("factor_pattern must have one row per item")
    k = len(P[0])
    if k < 1:
        raise ValueError("factor_pattern must have at least one column")
    mask = [[1 if abs(v) > 0.0 else 0 for v in r] for r in P]
    if all(sum(r) == 0 for r in mask):
        raise ValueError("factor_pattern frees no loading at all")
    lam, psi, fml, resid, it = _fa_em(S, mask)
    comm = [sum(lam[i][j] ** 2 for j in range(k)) for i in range(p)]
    tr = sum(S[i][i] for i in range(p))
    return RichResult(payload={
        "estimate": sum(comm) / tr,
        "loadings": lam,
        "uniquenesses": psi,
        "fml": fml,
        "max_resid": resid,
        "communality": comm,
        "n_iter": it,
        "p": p,
        "k": k,
        "method": "CFA multi-factor with cross-loadings allowed",
    })


def cheatsheet():
    return "cfafm2: CFA multi-factor with cross-loadings allowed"


# compact alias per ledger/NAMING.md
cfamultifactor = cfa_multifactor
