# morie.fn -- slice s04 (rootcoder007/morie)
"""Logistic regression log-likelihood.

Source consulted: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, Section 3.7.  The chapter models the response as
Bernoulli with p(x_i; beta) = exp(eta_i) / (1 + exp(eta_i)),
eta_i = beta_0 + x_i' beta_0, and writes the log-likelihood as

    l(beta; y) = sum_i y_i eta_i - sum_i log(1 + exp(eta_i)),

with gradient X'(y - p(X; beta)) and Hessian -X' W X, W = diag(p(1-p)).

The sum-of-logs form in the docstring formula and the chapter's form are
the same function; the chapter's form is used because it is the one that
does not lose precision when p underflows to 0 or overflows to 1, and
the log-sum-exp guard keeps it finite for |eta| of any size.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["logistic_log_likelihood"]


def _log1pexp(z):
    # log(1 + e^z) without overflow: for large z it is z + log(1 + e^-z).
    if z > 0.0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def logistic_log_likelihood(y, X, beta):
    """Log-likelihood, score and fitted probabilities of a logistic model.

    Parameters
    ----------
    y : array-like
        Binary response, entries 0 or 1.
    X : array-like
        n-by-p design matrix.  An intercept column is NOT added; include
        it in X if the model has one.
    beta : array-like
        Coefficient vector of length p.

    Returns
    -------
    estimate : the log-likelihood
    loglik   : the same value
    p        : the fitted probabilities
    gradient : X'(y - p), the score
    """
    yy = k.vec(y)
    XX = k.mat(X)
    bb = k.vec(beta)
    n = len(yy)
    if n == 0:
        raise ValueError("logistic_log_likelihood: y is empty")
    if len(XX) != n:
        raise ValueError("logistic_log_likelihood: X has a different number of rows than y")
    p_ = len(XX[0])
    if len(bb) != p_:
        raise ValueError("logistic_log_likelihood: beta does not match the columns of X")
    for v in yy:
        if v != 0.0 and v != 1.0:
            raise ValueError("logistic_log_likelihood: y must be 0 or 1")
    ll = 0.0
    p = []
    grad = [0.0] * p_
    for i in range(n):
        eta = 0.0
        for j in range(p_):
            eta += XX[i][j] * bb[j]
        ll += yy[i] * eta - _log1pexp(eta)
        pi = k.sigmoid(eta)
        p.append(pi)
        r = yy[i] - pi
        for j in range(p_):
            grad[j] += XX[i][j] * r
    return RichResult(
        title="Logistic log-likelihood",
        summary_lines=[("n", n), ("p", p_)],
        payload={
            "estimate": ll,
            "loglik": ll,
            "p": p,
            "gradient": grad,
            "n": n,
            "method": "l(beta;y) = sum y_i eta_i - sum log(1+exp(eta_i)), Chapter 3 Sect. 3.7",
        },
    )


def cheatsheet():
    return "lgobj: Logistic regression log-likelihood"
