# morie.fn -- slice s04 (rootcoder007/morie)
"""Negative binomial regression for overdispersed count data.

Book sections read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer.  Volume [Pages 35-70], Chapter 2, the generalised
linear model table on p. 40, lists the Counts / Negative binomial row
with the Log link and mean exp(X beta_hat), which is the link and mean
used here.  Volume [Pages 379-425], Chapter 10, Section 10.7.2, p. 401,
adds that "for count data the loss function can be obtained under a
negative binomial distribution, which can do a better job than the
Poisson distribution when the assumption of equal mean and variance is
hard to justify".

NOT IN THE BOOK.  All seventeen page-range volumes were searched: the
negative binomial appears only as a row in that table and as the remark
above.  The book never writes the mass function, never estimates the
dispersion, and its index ([Pages 683-691]) has no entry for it.  The
mass function and the moments used here are the ones the function's own
docstring states,

    P(Y=k) = C(k+r-1, k) p^r (1-p)^k,  E[Y] = mu,
    Var[Y] = mu + mu^2/r,

that is, the NB2 parameterisation; the dispersion is estimated from the
method-of-moments identity Var = mu + mu^2/r implied by those same two
moments, so nothing beyond the docstring's own statement is assumed.
The fit is Fisher scoring on the log link with the NB2 weight
w_i = mu_i / (1 + mu_i/r), alternated with that moment update for r.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["negative_binomial_dispersion"]


def negative_binomial_dispersion(y, X, link="log", max_iter=100, tol=1e-12):
    """NB2 regression: mean by Fisher scoring, dispersion by moments.

    Parameters
    ----------
    y : array-like
        Non-negative integer counts.
    X : array-like
        n-by-p design matrix; include an intercept column if wanted.
    link : str
        Only "log" is offered, the link the Chapter 2 table gives.

    Returns
    -------
    estimate : r_hat, the dispersion
    mu_hat   : the fitted means
    r_hat    : the same dispersion
    beta     : the coefficients on the log scale
    """
    yy = core.vec(y)
    n = len(yy)
    if n == 0:
        raise ValueError("negative_binomial_dispersion: y is empty")
    for v in yy:
        if v < 0.0 or v != math.floor(v):
            raise ValueError("negative_binomial_dispersion: y must be non-negative counts")
    XX = core.mat(X)
    if len(XX) != n:
        raise ValueError("negative_binomial_dispersion: X has a different number of rows than y")
    p = len(XX[0])
    if link != "log":
        raise ValueError("negative_binomial_dispersion: only the log link of the Chapter 2 table is offered")
    beta = [0.0] * p
    r = 1e6
    mu = [1.0] * n
    for _ in range(int(max_iter)):
        prev = list(beta)
        # Fisher scoring for the log link with NB2 weights
        A = [[0.0] * p for _ in range(p)]
        b = [0.0] * p
        for i in range(n):
            eta = 0.0
            for j in range(p):
                eta += XX[i][j] * beta[j]
            eta = min(max(eta, -300.0), 300.0)
            m = math.exp(eta)
            mu[i] = m
            w = m / (1.0 + m / r) if r > 0.0 else m
            z = eta + (yy[i] - m) / m if m > 0.0 else eta
            for a in range(p):
                b[a] += w * XX[i][a] * z
                for c in range(p):
                    A[a][c] += w * XX[i][a] * XX[i][c]
        beta = core.ridgesolve(A, b, 1e-12)
        # moment update of r from Var = mu + mu^2/r
        s2 = 0.0
        sm = 0.0
        for i in range(n):
            eta = 0.0
            for j in range(p):
                eta += XX[i][j] * beta[j]
            m = math.exp(min(max(eta, -300.0), 300.0))
            mu[i] = m
            s2 += (yy[i] - m) ** 2 - m
            sm += m * m
        r = sm / s2 if s2 > 0.0 else float("inf")
        d = 0.0
        for j in range(p):
            d = max(d, abs(beta[j] - prev[j]))
        if d < tol:
            break
    return RichResult(
        title="Negative binomial dispersion",
        summary_lines=[("n", n), ("p", p)],
        payload={
            "estimate": r,
            "mu_hat": mu,
            "r_hat": r,
            "beta": beta,
            "n": n,
            "method": "NB2 log-link mean (Chapter 2 GLM table) with r from Var = mu + mu^2/r",
        },
    )


def cheatsheet():
    return "nbdsp: Negative binomial regression for overdispersed count data"
