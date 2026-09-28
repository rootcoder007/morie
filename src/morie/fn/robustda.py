# morie.fn -- function file (rootcoder007/morie)
"""Robust linear discriminant analysis: class centres and the pooled within-class scatter taken from
FAST-MCD fits instead of sample means and covariances (Hawkins and McLachlan 1997)."""

from __future__ import annotations

import math

from ._qpcore import solve, ssum
from ._richresult import RichResult
from .fastm import fast_mcd

__all__ = ["robust_lda"]


def robust_lda(X, y, *, prior=None, newdata=None) -> RichResult:
    r"""Robust LDA with per-class MCD location and pooled MCD scatter.

    For each class ``k`` FAST-MCD (``morie.fn.fastm``) gives a centre
    ``m_k`` and a consistency-corrected scatter ``S_k`` on ``h_k``
    observations. The pooled scatter is ``W = sum_k h_k S_k / (sum_k h_k -
    g)`` (the "mcd-A" pooling of Todorov's ``rrcov::Linda``). The linear
    score is ``d_k(x) = m_k' W^-1 x - m_k' W^-1 m_k / 2 + log pi_k`` with
    ``pi_k`` the class proportions unless ``prior`` is given; each row of
    ``newdata`` (default ``X``) is assigned to the arg-max class. Returns
    ``classes``, ``centers``, ``pooled_cov``, ``prior``, ``scores``,
    ``predicted`` and the resubstitution ``apparent_error`` when
    ``newdata`` is ``None``.

    References
    ----------
    Hawkins, D. M. and McLachlan, G. J. (1997). High-breakdown linear
    discriminant analysis. *Journal of the American Statistical
    Association*, 92, 136-143.
    Todorov, V. and Filzmoser, P. (2009). An object-oriented framework for
    robust multivariate analysis. *Journal of Statistical Software*, 32(3).

    Examples
    --------
    >>> X = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.4], [9, 0], [0.2, 0.8],
    ...      [5, 5], [6, 5], [5, 6], [6, 6], [5.5, 5.4], [5.2, 5.8], [-9, 9]]
    >>> y = [0] * 7 + [1] * 7
    >>> robust_lda(X, y).predicted
    [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]
    """
    Xm = [[float(v) for v in r] for r in X]
    labels = list(y)
    if len(labels) != len(Xm):
        raise ValueError("X and y must have the same length")
    p = len(Xm[0])
    classes = sorted(set(labels))
    g = len(classes)
    if g < 2:
        raise ValueError("need at least two classes")
    centers, W, htot = [], [[0.0] * p for _ in range(p)], 0
    for c in classes:
        fit = fast_mcd([Xm[i] for i in range(len(Xm)) if labels[i] == c])
        centers.append([float(v) for v in fit.center])
        h = fit.h
        htot += h
        for a in range(p):
            for b in range(p):
                W[a][b] += h * fit.cov[a][b]
    W = [[W[a][b] / (htot - g) for b in range(p)] for a in range(p)]
    if prior is None:
        prior = [sum(1 for v in labels if v == c) / len(labels) for c in classes]
    prior = [float(v) for v in prior]
    coef = [solve(W, m) for m in centers]
    const = [-0.5 * ssum(coef[k][a] * centers[k][a] for a in range(p)) + math.log(prior[k]) for k in range(g)]
    Z = Xm if newdata is None else [[float(v) for v in r] for r in newdata]
    scores = [[ssum(coef[k][a] * z[a] for a in range(p)) + const[k] for k in range(g)] for z in Z]
    pred = [classes[max(range(g), key=lambda k, s=s: (s[k], -k))] for s in scores]
    out = {
        "classes": classes,
        "centers": centers,
        "pooled_cov": W,
        "prior": prior,
        "scores": scores,
        "predicted": pred,
    }
    if newdata is None:
        out["apparent_error"] = sum(1 for a, b in zip(pred, labels) if a != b) / len(labels)
    return RichResult(payload=out)


def cheatsheet() -> str:
    return "robust_lda(X, y) -> LDA with FAST-MCD class centres and pooled MCD scatter (Hawkins-McLachlan)."
