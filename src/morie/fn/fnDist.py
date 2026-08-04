# morie.fn -- function file (rootcoder007/morie)
"""L2 distance between two functional observations.

Ramsay and Silverman (2005), *Functional Data Analysis*, 2nd ed.,
Springer: the inner product on the function space is
<f, g> = integral f(t) g(t) dt and the induced metric is
d(f, g) = ||f - g|| = sqrt( integral (f(t) - g(t))^2 dt ), which is the
formula named in the stub docstring.

The integral is taken over the WHOLE observation interval by the
composite trapezoid rule.  This is deliberate and load-bearing: a
sibling module in this package once integrated over [a+h, b-h],
dropping both end intervals, and returned 3.8667 where the closed form
is 4.  The anchors below are closed forms over the whole interval.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["functional_distance"]


def _trapz(t, v):
    """Composite trapezoid rule over the whole of t."""
    s = 0.0
    for i in range(len(t) - 1):
        s += 0.5 * (v[i] + v[i + 1]) * (t[i + 1] - t[i])
    return s


def functional_distance(f, g, t=None):
    """d(f, g) = sqrt(integral (f - g)^2).

    Parameters
    ----------
    f, g : array-like
        The two curves, sampled at common points.
    t : array-like, optional
        The sampling grid.  Defaults to an equally spaced grid on [0, 1].

    Returns
    -------
    estimate : the L2 distance
    l2sq     : integral (f - g)^2
    l1       : integral |f - g|
    sup      : max |f - g|
    """
    ff = k.vec(f)
    gg = k.vec(g)
    n = len(ff)
    if n == 0:
        raise ValueError("functional_distance: f is empty")
    if len(gg) != n:
        raise ValueError("functional_distance: f and g must have the same length")
    if n < 2:
        raise ValueError("functional_distance: need at least two sampling points")
    if t is None:
        tt = [i / float(n - 1) for i in range(n)]
    else:
        tt = k.vec(t)
        if len(tt) != n:
            raise ValueError("functional_distance: t must match the curve length")
    d = [ff[i] - gg[i] for i in range(n)]
    sq = [x * x for x in d]
    ab = [abs(x) for x in d]
    l2sq = _trapz(tt, sq)
    return RichResult(
        title="Functional L2 distance",
        summary_lines=[("points", n), ("distance", math.sqrt(l2sq) if l2sq > 0 else 0.0)],
        payload={
            "estimate": math.sqrt(l2sq) if l2sq > 0.0 else 0.0,
            "l2sq": l2sq,
            "l1": _trapz(tt, ab),
            "sup": max(ab),
            "n": n,
            "method": "Ramsay-Silverman (2005) L2 metric, composite trapezoid over the whole interval",
        },
    )


def cheatsheet():
    return "fnDist: L2 distance between two curves"
