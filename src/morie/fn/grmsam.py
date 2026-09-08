# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Samejima's graded response model.

Samejima (1969), "Estimation of latent ability using a response
pattern of graded scores", Psychometrika Monograph Supplement 34(4,
Pt. 2), doi:10.1007/BF03372160.  The model is stated through the
cumulative ("operating") characteristic functions

    P*_k(theta) = 1 / (1 + exp(-a (theta - b_k))),   k = 1 .. m,
    P*_0 = 1,   P*_{m+1} = 0,
    P_k(theta) = P*_k(theta) - P*_{k+1}(theta),

so the thresholds b_k must increase, otherwise a category probability
goes negative -- the constraint is checked rather than assumed.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["graded_response_samejima"]


def _grm_probs(theta, a, b):
    m = len(b)
    star = [1.0] + [core.sigmoid(a * (theta - b[k])) for k in range(m)] + [0.0]
    return [star[k] - star[k + 1] for k in range(m + 1)]


def graded_response_samejima(y, theta, a, b_k):
    """Graded category probabilities and the observed-response likelihood.

    Parameters
    ----------
    y : array-like of int
        Observed categories, 0-based (0 .. m).
    theta : array-like
        Person abilities, same length as y.
    a : float
        Item slope.
    b_k : array-like
        The m strictly increasing category thresholds.
    """
    ys = [int(v) for v in core.vec(y)]
    th = core.vec(theta)
    b = core.vec(b_k)
    if len(ys) == 0:
        raise ValueError("graded_response_samejima: y is empty")
    if len(th) != len(ys):
        raise ValueError("graded_response_samejima: y and theta have different lengths")
    if len(b) == 0:
        raise ValueError("graded_response_samejima: b_k is empty")
    for k in range(1, len(b)):
        if b[k] <= b[k - 1]:
            raise ValueError("graded_response_samejima: thresholds must strictly increase")
    av = float(a)
    if av <= 0:
        raise ValueError("graded_response_samejima: a must be positive")
    m = len(b)
    ll = 0.0
    pobs = []
    for i in range(len(ys)):
        if ys[i] < 0 or ys[i] > m:
            raise ValueError("graded_response_samejima: response outside the category range")
        p = _grm_probs(th[i], av, b)
        pobs.append(p[ys[i]])
        ll += math.log(p[ys[i]])
    first = _grm_probs(th[0], av, b)
    tot = 0.0
    for v in pobs:
        tot += v
    return RichResult(
        title="Graded response model",
        summary_lines=[("n", len(ys)), ("categories", m + 1)],
        payload={
            "estimate": tot / len(ys),
            "p_observed": pobs,
            "probs_first": first,
            "loglik": ll,
            "categories": m + 1,
            "n": len(ys),
            "method": "P_k = P*_k - P*_{k+1} with logistic P*, Samejima (1969)",
        },
    )


def cheatsheet():
    return "grmsam: Samejima graded response model"
