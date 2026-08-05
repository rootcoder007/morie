# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Hoeffding's inequality for bounded independent variables.

Hoeffding (1963), "Probability inequalities for sums of bounded random
variables", JASA 58(301):13-30, doi:10.1080/01621459.1963.10500830,
Theorem 2.  For independent X_i taking values in [a, b],

    P( |Sbar_n - E Sbar_n| >= t ) <= 2 exp( -2 n t^2 / (b - a)^2 ).

The bound is distribution-free, which is its point; it is vacuous
(at least 1) until t exceeds (b - a) sqrt(ln 2 / (2n)), and that
threshold is the independent criterion the tests check against.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["hoeffding_inequality"]


def hoeffding_inequality(a, b, n, t):
    """Two-sided Hoeffding tail bound and the sample size it implies.

    Parameters
    ----------
    a, b : floats with a < b, the support of each variable.
    n : int, sample size.
    t : float >= 0, the deviation of the mean from its expectation.
    """
    av = float(a)
    bv = float(b)
    if not bv > av:
        raise ValueError("hoeffding_inequality: need a < b")
    nn = int(n)
    if nn < 1:
        raise ValueError("hoeffding_inequality: n must be at least 1")
    tv = float(t)
    if tv < 0:
        raise ValueError("hoeffding_inequality: t must be non-negative")
    rng = bv - av
    ex = -2.0 * nn * tv * tv / (rng * rng)
    bound = 2.0 * math.exp(ex)
    one_sided = math.exp(ex)
    informative = 1 if bound < 1.0 else 0
    t_min = rng * math.sqrt(math.log(2.0) / (2.0 * nn))
    return RichResult(
        title="Hoeffding inequality",
        summary_lines=[("n", nn), ("t", tv)],
        payload={
            "estimate": min(bound, 1.0),
            "bound": bound,
            "one_sided": one_sided,
            "informative": informative,
            "t_min": t_min,
            "n": nn,
            "method": "2 exp(-2 n t^2 / (b - a)^2), Hoeffding (1963) Theorem 2",
        },
    )


def cheatsheet():
    return "hffdsg: Hoeffding's inequality"
