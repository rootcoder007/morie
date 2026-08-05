# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Graded response model over a response matrix.

Samejima (1969), Psychometrika Monograph Supplement 34(4, Pt. 2),
doi:10.1007/BF03372160.

The model itself lives in ``morie.fn.grmsam``; the wave2 audit flagged
this module as a duplicate of it and it is one, so the category
probabilities are NOT recomputed here.  This module adds the matrix
interface only.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from .grmsam import graded_response_samejima as _grm

from ._richresult import RichResult

__all__ = ["graded_response"]


def graded_response(X, ncats, theta=None, a=None, b=None):
    """Score a response matrix under the graded response model.

    Parameters
    ----------
    X : n x J matrix of 0-based category responses (0 .. ncats-1).
    ncats : int, number of categories; each item has ncats-1 thresholds.
    theta : length-n abilities; zeros by default.
    a : length-J slopes; ones by default.
    b : J x (ncats-1) increasing thresholds; equally spaced by default.
    """
    M = core.mat(X)
    n = len(M)
    if n == 0:
        raise ValueError("graded_response: X is empty")
    J = len(M[0])
    K = int(ncats)
    if K < 2:
        raise ValueError("graded_response: need at least two categories")
    th = [0.0] * n if theta is None else core.vec(theta)
    if len(th) != n:
        raise ValueError("graded_response: theta and X have different lengths")
    av = [1.0] * J if a is None else core.vec(a)
    if len(av) != J:
        raise ValueError("graded_response: a must have one slope per item")
    if b is None:
        bm = [[(k - (K - 1) / 2.0) for k in range(1, K)] for _ in range(J)]
    else:
        bm = core.mat(b)
        if len(bm) != J or len(bm[0]) != K - 1:
            raise ValueError("graded_response: b must be J x (ncats - 1)")
    ll = 0.0
    per_item = []
    for j in range(J):
        col = [int(M[i][j]) for i in range(n)]
        r = _grm(col, th, av[j], bm[j])
        per_item.append(r["loglik"])
        ll += r["loglik"]
    return RichResult(
        title="Graded response model over a response matrix",
        summary_lines=[("persons", n), ("items", J), ("categories", K)],
        payload={
            "estimate": ll,
            "loglik": ll,
            "loglik_item": per_item,
            "categories": K,
            "n": n,
            "method": "per-item GRM likelihood, delegating to grmsam; Samejima (1969)",
        },
    )


def cheatsheet():
    return "irtgrm: graded response model over a response matrix (delegates to grmsam)"


# compact alias per ledger/NAMING.md
gradedresponse = graded_response
