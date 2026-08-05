# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Four-parameter logistic item response model.

The four-parameter logistic adds an upper asymptote to Birnbaum's
three-parameter model; see Lord (1980), *Applications of Item Response
Theory to Practical Testing Problems*, Lawrence Erlbaum, chapter 2 for
the three-parameter form and Barton and Lord (1981), ETS Research
Report RR-81-20, doi:10.1002/j.2333-8504.1981.tb01255.x, for the
upper-asymptote extension:

    P(theta) = c + (d - c) / (1 + exp(-a (theta - b))).

c is the lower asymptote (guessing), d the upper asymptote (slipping);
c = 0, d = 1 recovers the two-parameter logistic exactly.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["four_parameter_logistic"]


def four_parameter_logistic(y, theta, a, b, c=0.0, d=1.0):
    """4PL success probabilities and log-likelihood."""
    ys = [int(v) for v in core.vec(y)]
    th = core.vec(theta)
    if len(ys) == 0:
        raise ValueError("four_parameter_logistic: y is empty")
    if len(th) != len(ys):
        raise ValueError("four_parameter_logistic: y and theta have different lengths")
    for v in ys:
        if v not in (0, 1):
            raise ValueError("four_parameter_logistic: responses must be 0 or 1")
    av, bv, cv, dv = float(a), float(b), float(c), float(d)
    if av <= 0:
        raise ValueError("four_parameter_logistic: a must be positive")
    if cv < 0 or dv > 1 or cv >= dv:
        raise ValueError("four_parameter_logistic: need 0 <= c < d <= 1")
    p = [cv + (dv - cv) * core.sigmoid(av * (th[i] - bv)) for i in range(len(ys))]
    ll = 0.0
    for i in range(len(ys)):
        ll += math.log(p[i]) if ys[i] == 1 else math.log(1.0 - p[i])
    tot = 0.0
    for v in p:
        tot += v
    return RichResult(
        title="Four-parameter logistic model",
        summary_lines=[("n", len(ys)), ("a", av), ("b", bv)],
        payload={
            "estimate": tot / len(ys),
            "p": p,
            "loglik": ll,
            "a": av,
            "b": bv,
            "c": cv,
            "d": dv,
            "n": len(ys),
            "method": "P = c + (d - c)/(1 + exp(-a(theta - b)))",
        },
    )


def cheatsheet():
    return "irt4pl: four-parameter logistic model"
