# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Generalized partial credit model over a response matrix.

Muraki (1992), Applied Psychological Measurement 16(2):159-176,
doi:10.1177/014662169201600206.

The model itself lives in ``morie.fn.gpcm``; the wave2 audit flagged
this module as a duplicate of it and it is one, so the category
probabilities are NOT recomputed here.  What this module adds is the
matrix interface: a persons-by-items response matrix scored against
per-item parameters, returning the per-item log-likelihood and the
total.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from .gpcm import generalized_partial_credit as _gpcm

from ._richresult import RichResult

__all__ = ["generalized_partial_credit"]


def generalized_partial_credit(X, ncats, theta=None, a=None, b=None):
    """Score a response matrix under the GPCM.

    Parameters
    ----------
    X : n x J matrix of 0-based category responses.
    ncats : int, the number of categories (b has ncats entries per item).
    theta : length-n abilities; zeros by default.
    a : length-J slopes; ones by default.
    b : J x ncats step parameters; equally spaced by default.
    """
    M = core.mat(X)
    n = len(M)
    if n == 0:
        raise ValueError("generalized_partial_credit: X is empty")
    J = len(M[0])
    K = int(ncats)
    if K < 2:
        raise ValueError("generalized_partial_credit: need at least two categories")
    th = [0.0] * n if theta is None else core.vec(theta)
    if len(th) != n:
        raise ValueError("generalized_partial_credit: theta and X have different lengths")
    av = [1.0] * J if a is None else core.vec(a)
    if len(av) != J:
        raise ValueError("generalized_partial_credit: a must have one slope per item")
    if b is None:
        bm = [[0.0] + [(k - (K - 1) / 2.0) for k in range(1, K)] for _ in range(J)]
    else:
        bm = core.mat(b)
        if len(bm) != J or len(bm[0]) != K:
            raise ValueError("generalized_partial_credit: b must be J x ncats")
    ll = 0.0
    per_item = []
    for j in range(J):
        col = [int(M[i][j]) for i in range(n)]
        r = _gpcm(col, th, av[j], bm[j])
        per_item.append(r["loglik"])
        ll += r["loglik"]
    return RichResult(
        title="GPCM over a response matrix",
        summary_lines=[("persons", n), ("items", J), ("categories", K)],
        payload={
            "estimate": ll,
            "loglik": ll,
            "loglik_item": per_item,
            "categories": K,
            "n": n,
            "method": "per-item GPCM likelihood, delegating to gpcm; Muraki (1992)",
        },
    )


def cheatsheet():
    return "irtgpc: GPCM over a response matrix (delegates to gpcm)"
