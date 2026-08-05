# morie.fn -- function file (rootcoder007/morie)
"""Single-layer feed-forward network as an adaptive basis expansion.

Hastie, Tibshirani and Friedman (2009), *The Elements of Statistical
Learning*, 2nd ed., Springer, Section 2.8.3, book p. 36 (PDF p. 55):

    f_theta(x) = sum_{m=1..M} beta_m sigma(alpha_m' x + b_m)        (2.45)

where "sigma(x) = 1/(1 + e^-x) is known as the activation function".
The book is explicit that with the directions alpha_m and biases b_m
fixed this is a linear basis expansion, so given y the output weights
beta are obtained in closed form by least squares on the hidden
activations; only alpha and b would require search, and they are inputs
here rather than being invented.
"""

from __future__ import annotations

from . import _s03core as k

from ._richresult import RichResult

__all__ = ["nnet1lay"]


def nnet1lay(X, alpha, b, beta=None, y=None):
    """Equation (2.45).

    Parameters
    ----------
    X : array-like
        N-by-p design.
    alpha : array-like
        p-by-M matrix whose columns are the directions alpha_m.
    b : array-like
        M-vector of bias terms b_m.
    beta : array-like, optional
        M-vector of output weights; required unless y is given.
    y : array-like, optional
        N-vector; when given, beta is refitted by least squares.

    Returns
    -------
    RichResult with keys estimate, fitted, hidden, beta, rss, n, p, M,
    method.
    """
    Xm = k.mat(X)
    Am = k.mat(alpha)
    bv = k.vec(b)
    n = k.nrow(Xm)
    if n == 0:
        raise ValueError("nnet1lay: X is empty")
    p = k.ncol(Xm)
    if k.nrow(Am) != p:
        raise ValueError("nnet1lay: alpha must have one row per column of X")
    M = k.ncol(Am)
    if M == 0:
        raise ValueError("nnet1lay: alpha has no columns")
    if len(bv) != M:
        raise ValueError("nnet1lay: b must have one entry per hidden unit")
    Z = [[k.sigmoid(sum(Xm[i][a] * Am[a][m] for a in range(p)) + bv[m]) for m in range(M)] for i in range(n)]
    if y is not None:
        yv = k.vec(y)
        if len(yv) != n:
            raise ValueError("nnet1lay: X and y must have the same number of rows")
        if n < M:
            raise ValueError("nnet1lay: fewer observations than hidden units")
        bw = k.lstsq(Z, yv, 0.0)
    elif beta is not None:
        bw = k.vec(beta)
        if len(bw) != M:
            raise ValueError("nnet1lay: beta must have one entry per hidden unit")
    else:
        raise ValueError("nnet1lay: supply beta, or y to fit it")
    fitted = [sum(Z[i][m] * bw[m] for m in range(M)) for i in range(n)]
    rss = float("nan")
    if y is not None:
        yv = k.vec(y)
        rss = sum((yv[i] - fitted[i]) ** 2 for i in range(n))
    return RichResult(
        title="Single-layer network, ESL eq. (2.45)",
        summary_lines=[("n", n), ("M", M), ("rss", rss)],
        payload={
            "estimate": fitted[0],
            "fitted": fitted,
            "hidden": Z,
            "beta": bw,
            "rss": rss,
            "n": n,
            "p": p,
            "M": M,
            "method": "Hastie-Tibshirani-Friedman (2009) ESL eq. (2.45)",
        },
    )


def cheatsheet():
    return "nnet1lay: f(x) = sum_m beta_m sigma(alpha_m'x + b_m), ESL eq. (2.45)"
