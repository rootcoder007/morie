# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Rasch (one-parameter logistic) model.

Rasch (1960), *Probabilistic Models for Some Intelligence and
Attainment Tests*, Danmarks Paedagogiske Institut, Copenhagen; the
dichotomous model

    P(X = 1 | theta, b) = exp(theta - b) / (1 + exp(theta - b)).

The single item parameter is what gives the model its defining
property: the raw score is a sufficient statistic for theta, which the
tests check directly (two persons with the same total score get the
same likelihood profile).
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["rasch_one_parameter"]


def rasch_one_parameter(y, theta, b):
    """Rasch success probabilities and log-likelihood.

    Parameters
    ----------
    y : array-like of {0, 1}
        Observed responses.
    theta : array-like
        Person abilities, same length as y.
    b : float
        Item difficulty.
    """
    ys = [int(v) for v in core.vec(y)]
    th = core.vec(theta)
    if len(ys) == 0:
        raise ValueError("rasch_one_parameter: y is empty")
    if len(th) != len(ys):
        raise ValueError("rasch_one_parameter: y and theta have different lengths")
    for v in ys:
        if v not in (0, 1):
            raise ValueError("rasch_one_parameter: responses must be 0 or 1")
    bv = float(b)
    p = [core.sigmoid(th[i] - bv) for i in range(len(ys))]
    ll = 0.0
    for i in range(len(ys)):
        ll += math.log(p[i]) if ys[i] == 1 else math.log(1.0 - p[i])
    info = [p[i] * (1.0 - p[i]) for i in range(len(ys))]
    tot = 0.0
    for v in p:
        tot += v
    return RichResult(
        title="Rasch model",
        summary_lines=[("n", len(ys)), ("b", bv)],
        payload={
            "estimate": tot / len(ys),
            "p": p,
            "information": info,
            "loglik": ll,
            "b": bv,
            "n": len(ys),
            "method": "P = exp(theta - b)/(1 + exp(theta - b)), Rasch (1960)",
        },
    )


def cheatsheet():
    return "irt1pl: Rasch one-parameter logistic model"
