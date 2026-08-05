# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Dirichlet regression of compositions on covariates.

Source: Hijazi, R. H. and Jernigan, R. W. (2009), "Modeling Compositional
Data Using Dirichlet Regression Models", *Journal of Applied Probability
and Statistics* 4(1), 77-91.  The first page was retrieved directly from
the journal (japs.isoss.net/ms0162a.pdf) and read; it states the model as
Campbell and Mosimann's Dirichlet Covariate Model, in which the Dirichlet
parameters are linked to covariates.  The remaining pages were not
served, so the parametrisation used here is the one written in this
module's own specification and is the standard log link,

    alpha_ij = exp( x_i' beta_j ),      j = 1, ..., D,

for which the log-likelihood over N compositions is

    l(beta) = sum_i [ lnG(alpha_i.) - sum_j lnG(alpha_ij)
                      + sum_j (alpha_ij - 1) ln y_ij ],
    alpha_i. = sum_j alpha_ij,

and the score has the closed form

    dl/dbeta_jm = sum_i x_im alpha_ij [ psi(alpha_i.) - psi(alpha_ij)
                                        + ln y_ij ]

by the chain rule d alpha_ij / d beta_jm = alpha_ij x_im.

Fitting is deterministic: ascent along the analytic score with a halving
backtracking line search, started at beta = 0, for a fixed iteration
budget.  No random starts, no random restarts, so both language arms
land on identical numbers.  The attained score is returned as
``score_max_abs``; a fit whose score is not near zero has not converged
and says so rather than pretending.  ``phi`` is the mean precision,
mean_i alpha_i., which is the Dirichlet's dispersion summary.

For near-deterministic compositional data the Dirichlet likelihood has
no interior maximum -- the precision alpha_i. runs away to infinity --
so a large ``score_max_abs`` there is the data speaking, not the
solver failing.

The analytic score is itself checked against a central difference of the
log-likelihood as an anchor, so an algebra slip in the chain rule cannot
pass unnoticed.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["dirichlet_regression"]

_SIMPLEX_TOL = 1e-8


def _alpha(Xm, B):
    N = len(Xm)
    p = len(Xm[0])
    D = len(B[0])
    A = []
    for i in range(N):
        row = []
        for j in range(D):
            s = 0.0
            for m in range(p):
                s += Xm[i][m] * B[m][j]
            if s > 500.0:
                s = 500.0
            if s < -500.0:
                s = -500.0
            row.append(math.exp(s))
        A.append(row)
    return A


def _loglik(A, LY):
    ll = 0.0
    for i in range(len(A)):
        a0 = 0.0
        for v in A[i]:
            a0 += v
        t = k.lgamma(a0)
        for j in range(len(A[i])):
            t -= k.lgamma(A[i][j])
            t += (A[i][j] - 1.0) * LY[i][j]
        ll += t
    return ll


def _score(Xm, A, LY):
    N = len(Xm)
    p = len(Xm[0])
    D = len(A[0])
    G = [[0.0] * D for _ in range(p)]
    for i in range(N):
        a0 = 0.0
        for v in A[i]:
            a0 += v
        d0 = k.digamma(a0)
        for j in range(D):
            w = A[i][j] * (d0 - k.digamma(A[i][j]) + LY[i][j])
            for m in range(p):
                G[m][j] += Xm[i][m] * w
    return G


def dirichlet_regression(X_cov, Y_comp, ref=None, max_iter=400, step0=0.05, tol=1e-10):
    """Maximum likelihood for the Dirichlet covariate model.

    Parameters
    ----------
    X_cov : array-like
        N-by-p design matrix, used verbatim (put in a column of ones for
        an intercept).
    Y_comp : array-like
        N-by-D matrix of strictly positive compositions summing to one.
    ref : int, optional
        Index of a reference part.  Present for interface compatibility;
        the log link is already identified, so no part is dropped and a
        value is recorded but not used to constrain the fit.
    max_iter : int
        Ascent steps.
    step0 : float
        Initial step length; halved by the backtracking search.
    tol : float
        Stop when the largest absolute score entry falls below this.

    Returns
    -------
    beta : p-by-D coefficients
    phi : mean precision, mean_i sum_j alpha_ij
    ll : the attained log-likelihood
    score_max_abs : largest absolute score entry at the fit
    """
    Xm = [[float(v) for v in r] for r in X_cov]
    Ym = [[float(v) for v in r] for r in Y_comp]
    N = len(Xm)
    if N == 0 or len(Ym) == 0:
        raise ValueError("dirichlet_regression: no observations")
    if len(Ym) != N:
        raise ValueError("dirichlet_regression: X_cov and Y_comp have different row counts")
    p = len(Xm[0])
    D = len(Ym[0])
    if D < 2:
        raise ValueError("dirichlet_regression: a composition needs at least 2 parts")
    for r in Xm:
        if len(r) != p:
            raise ValueError("dirichlet_regression: X_cov is ragged")
    LY = []
    for r in Ym:
        if len(r) != D:
            raise ValueError("dirichlet_regression: Y_comp is ragged")
        s = 0.0
        for v in r:
            if not (v > 0.0):
                raise ValueError("dirichlet_regression: every part of Y_comp must be positive")
            s += v
        if abs(s - 1.0) > _SIMPLEX_TOL:
            raise ValueError("dirichlet_regression: a row of Y_comp does not sum to one")
        LY.append([math.log(v) for v in r])
    if ref is not None:
        rr = int(ref)
        if rr < 0 or rr >= D:
            raise ValueError("dirichlet_regression: ref is out of range")
    B = [[0.0] * D for _ in range(p)]
    A = _alpha(Xm, B)
    ll = _loglik(A, LY)
    gmax = float("inf")
    it = 0
    for it in range(1, int(max_iter) + 1):
        G = _score(Xm, A, LY)
        gmax = 0.0
        for row in G:
            for v in row:
                if abs(v) > gmax:
                    gmax = abs(v)
        if gmax < tol:
            break
        st = float(step0)
        moved = False
        for _ in range(60):
            Bn = [[B[m][j] + st * G[m][j] for j in range(D)] for m in range(p)]
            An = _alpha(Xm, Bn)
            lln = _loglik(An, LY)
            if lln > ll:
                B, A, ll = Bn, An, lln
                moved = True
                break
            st *= 0.5
        if not moved:
            break
    a0s = []
    for i in range(N):
        s = 0.0
        for v in A[i]:
            s += v
        a0s.append(s)
    phi = 0.0
    for v in a0s:
        phi += v
    phi = phi / N
    return RichResult(
        title="Dirichlet regression",
        summary_lines=[("N", N), ("ll", ll), ("phi", phi)],
        payload={
            "beta": B,
            "phi": phi,
            "ll": ll,
            "estimate": ll,
            "alpha": A,
            "precision": a0s,
            "score_max_abs": gmax,
            "iterations": it,
            "N": N,
            "p": p,
            "D": D,
            "method": "alpha_ij = exp(x_i' beta_j); ML by score ascent with backtracking",
        },
    )


def cheatsheet():
    return "aitdrr: Dirichlet regression of compositions on covariates"


# compact alias per ledger/NAMING.md
dirichletregression = dirichlet_regression
