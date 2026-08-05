# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Generalized partial credit model.

Muraki (1992), "A generalized partial credit model: application of an
EM algorithm", Applied Psychological Measurement 16(2):159-176,
doi:10.1177/014662169201600206, equation (1):

    P_jk(theta) = exp( sum_{v=0}^{k} a_j (theta - b_jv) )
                  / sum_{c=0}^{m_j} exp( sum_{v=0}^{c} a_j (theta - b_jv) ).

The v = 0 term is common to every numerator and cancels, which is why
b_j0 is arbitrary; it is kept here so the printed formula is followed
literally.  a_j is the slope, b_jv the step difficulties.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["generalized_partial_credit"]


def _gpcm_probs(theta, a, b):
    m = len(b)
    z = []
    run = 0.0
    for v in range(m):
        run += a * (theta - b[v])
        z.append(run)
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    tot = sum(e)
    return [v / tot for v in e]


def generalized_partial_credit(y, theta, a, b_j):
    """Category probabilities and the likelihood of the observed responses.

    Parameters
    ----------
    y : array-like of int
        Observed categories, 0-based, one per person.
    theta : array-like
        Person abilities, same length as y.
    a : float
        Item slope.
    b_j : array-like
        Step parameters b_j0 .. b_jm; their count sets the category count.
    """
    ys = [int(v) for v in core.vec(y)]
    th = core.vec(theta)
    b = core.vec(b_j)
    if len(ys) == 0:
        raise ValueError("generalized_partial_credit: y is empty")
    if len(th) != len(ys):
        raise ValueError("generalized_partial_credit: y and theta have different lengths")
    if len(b) < 2:
        raise ValueError("generalized_partial_credit: b_j needs at least two categories")
    av = float(a)
    if av <= 0:
        raise ValueError("generalized_partial_credit: a must be positive")
    m = len(b)
    ll = 0.0
    pobs = []
    for i in range(len(ys)):
        if ys[i] < 0 or ys[i] >= m:
            raise ValueError("generalized_partial_credit: response outside the category range")
        p = _gpcm_probs(th[i], av, b)
        pobs.append(p[ys[i]])
        ll += math.log(p[ys[i]])
    first = _gpcm_probs(th[0], av, b)
    tot = 0.0
    for v in pobs:
        tot += v
    return RichResult(
        title="Generalized partial credit model",
        summary_lines=[("n", len(ys)), ("categories", m)],
        payload={
            "estimate": tot / len(ys),
            "p_observed": pobs,
            "probs_first": first,
            "loglik": ll,
            "categories": m,
            "n": len(ys),
            "method": "GPCM eq. (1) of Muraki (1992)",
        },
    )


def cheatsheet():
    return "gpcm: generalized partial credit model"
