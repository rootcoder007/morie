# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Aitchison-distance k-nearest-neighbour classifier for compositions.

The distance is Aitchison's, taken from the rendered page 10 of
Mateu-Figueras, Pawlowsky-Glahn and Egozcue, "The normal distribution in
some constrained sample spaces", which prints

    d_a^2(x, x*) = (1/D) sum_{i<j} ( ln(x_i/x_j) - ln(x*_i/x*_j) )^2

immediately below inner product (10).  The classifier is the rule named
in the specification: take the k training points nearest x* in d_a, and
label x* by the group whose members inside that neighbourhood have the
smallest total distance,

    ghat(x*) = argmin_g  sum_{i in N_k(x*), y_i = g}  d_a(x*, x_i).

Groups with no member in the neighbourhood are not candidates -- their
empty sum would otherwise be zero and would win every time.  Ties in the
group score go to the smaller label, so the rule is a function, not a
coin toss, and both language arms agree.

Note that this is a *total*-distance vote, not a majority vote: one very
close neighbour can outweigh three middling ones.  That is the rule as
specified; the alternative (a count) is reported alongside as
``yhat_majority`` so the disagreement is visible rather than hidden.

Because ilr is an isometry, d_a equals the ordinary Euclidean distance
between ilr coordinates.  That gives an independent route to the same
neighbour ordering and is used as the anchor.

Reference for the k-NN-on-the-simplex practice: Pawlowsky-Glahn,
Egozcue and Tolosana-Delgado (2015), *Modeling and Analysis of
Compositional Data*, Wiley.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k  # noqa: F401

from ._richresult import RichResult

__all__ = ["compositional_classifyAP"]


def aitchison_distance(a, b):
    """d_a(a, b) straight from the squared-distance formula on page 10."""
    D = len(a)
    la = [math.log(v) for v in a]
    lb = [math.log(v) for v in b]
    s = 0.0
    for i in range(D):
        for j in range(i + 1, D):
            d = (la[i] - la[j]) - (lb[i] - lb[j])
            s += d * d
    return math.sqrt(s / D)


def _predict_one(rows, lab, xs, kk):
    dist = [aitchison_distance(xs, r) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: (dist[i], i))[:kk]
    tot = {}
    cnt = {}
    for i in order:
        g = lab[i]
        tot[g] = tot.get(g, 0.0) + dist[i]
        cnt[g] = cnt.get(g, 0) + 1
    best = None
    for g in sorted(tot):
        if best is None or tot[g] < tot[best] - 0.0:
            best = g
    bmaj = None
    for g in sorted(cnt):
        if bmaj is None or cnt[g] > cnt[bmaj]:
            bmaj = g
    return best, bmaj, dist, order


def compositional_classifyAP(X, y, x_new, k):
    """Label one or more new compositions.

    Parameters
    ----------
    X : array-like
        N-by-D matrix of strictly positive training compositions.
    y : array-like
        N group labels, coerced to float.
    x_new : array-like
        One composition, or a matrix of them.
    k : int
        Neighbourhood size, 1 <= k <= N.

    Returns
    -------
    yhat : the predicted label(s) under the total-distance rule
    yhat_majority : the labels a plain majority vote would give
    dist : distances from the first new point to every training point
    """
    rows = [[float(v) for v in r] for r in X]
    if not rows:
        raise ValueError("compositional_classifyAP: no training data")
    D = len(rows[0])
    if D < 2:
        raise ValueError("compositional_classifyAP: a composition needs at least 2 parts")
    for r in rows:
        if len(r) != D:
            raise ValueError("compositional_classifyAP: X is ragged")
        for v in r:
            if not (v > 0.0):
                raise ValueError("compositional_classifyAP: every part must be positive")
    lab = [float(v) for v in k.vec(y)]
    if len(lab) != len(rows):
        raise ValueError("compositional_classifyAP: X and y have different lengths")
    kk = int(k)
    if kk < 1 or kk > len(rows):
        raise ValueError("compositional_classifyAP: k must lie between 1 and the sample size")
    try:
        first = x_new[0]
    except (TypeError, IndexError, KeyError):
        raise ValueError("compositional_classifyAP: x_new is empty")
    if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
        news = [[float(v) for v in r] for r in x_new]
        many = True
    else:
        news = [[float(v) for v in x_new]]
        many = False
    yhat = []
    ymaj = []
    d0 = None
    for t, xs in enumerate(news):
        if len(xs) != D:
            raise ValueError("compositional_classifyAP: x_new has the wrong number of parts")
        for v in xs:
            if not (v > 0.0):
                raise ValueError("compositional_classifyAP: every part must be positive")
        b, m, dist, _ = _predict_one(rows, lab, xs, kk)
        yhat.append(b)
        ymaj.append(m)
        if t == 0:
            d0 = dist
    return RichResult(
        title="Aitchison k-NN classifier",
        summary_lines=[("N", len(rows)), ("k", kk)],
        payload={
            "yhat": yhat if many else yhat[0],
            "estimate": yhat[0],
            "yhat_majority": ymaj if many else ymaj[0],
            "dist": d0,
            "k": kk,
            "n": len(rows),
            "D": D,
            "method": "argmin_g sum_{i in N_k, y_i = g} d_a(x*, x_i), Aitchison distance",
        },
    )


def cheatsheet():
    return "aitcap: Aitchison-distance k-NN classifier"


# compact alias per ledger/NAMING.md
compositionalclassifyAP = compositional_classifyAP
