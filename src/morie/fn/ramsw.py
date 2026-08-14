# morie.fn -- function file (rootcoder007/morie)
"""Ramsay's exponential weight function and the E-type M-estimate.

Ramsay, J. O. (1977), "A comparative study of several robust estimates
of slope, intercept, and scale in linear regression", *Journal of the
American Statistical Association* 72(359), 608-615.  The weight
function proposed there is the exponential

    w(r) = exp(-a |r|),

which is the formula named in the stub docstring.  Unlike Huber's or
Tukey's, it is strictly positive everywhere: no observation is ever
given zero weight, it is only downweighted, and the downweighting is
governed by the single tuning constant a.

The location estimate is the fixed point of the weighted mean under
these weights: residuals are scaled by the median absolute deviation,
weights are recomputed, and the weighted mean is iterated to
convergence.  This is the ordinary iteratively reweighted least
squares loop with Ramsay's w in place of Huber's, run to a fixed
tolerance from a median start so that both language arms take the same
number of steps and land on the same number.

The tuning constant a = 0 turns every weight into exp(0) = 1, so the
estimate is then exactly the arithmetic mean.  That degenerate case is
this module's anchor: it is a closed form that does not depend on the
iteration at all.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["ramsay_weight"]


def ramsay_weight(y, a, max_iter=100, tol=1e-13):
    """Ramsay (1977) weights and the location estimate they define.

    Parameters
    ----------
    y : array-like
        The sample.
    a : float
        The tuning constant; a = 0 gives unit weights.
    max_iter : int
        Cap on the reweighting iterations.
    tol : float
        Convergence tolerance on the location.

    Returns
    -------
    estimate : the weighted location
    weights  : w(r_i) at the solution
    scale    : the MAD scale used
    iters    : iterations taken
    """
    v = k.vec(y)
    n = len(v)
    if n == 0:
        raise ValueError("ramsay_weight: y is empty")
    aa = float(a)
    if aa < 0.0:
        raise ValueError("ramsay_weight: the tuning constant must be non-negative")
    s = k.mad(v)
    if s <= 0.0:
        s = 1.0
    mu = k.median(v)
    it = 0
    w = [1.0] * n
    for it in range(1, int(max_iter) + 1):
        w = [math.exp(-aa * abs((v[i] - mu) / s)) for i in range(n)]
        sw = 0.0
        sx = 0.0
        for i in range(n):
            sw += w[i]
            sx += w[i] * v[i]
        new = sx / sw if sw > 0.0 else mu
        if abs(new - mu) <= tol:
            mu = new
            break
        mu = new
    return RichResult(
        title="Ramsay exponential-weight M-estimate",
        summary_lines=[("n", n), ("a", aa), ("estimate", mu), ("iterations", it)],
        payload={
            "estimate": mu,
            "weights": w,
            "scale": s,
            "a": aa,
            "iters": it,
            "n": n,
            "method": "Ramsay (1977) w(r) = exp(-a |r|), IRLS location from a median start with MAD scale",
        },
    )


def cheatsheet():
    return "ramsw: Ramsay (1977) exponential weight function"

# public names resolved by fn/_lazy_map.json
ramsayweight = ramsay_weight
