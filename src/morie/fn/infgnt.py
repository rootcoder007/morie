# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Fisher information metric of a parametric family.

Amari (1985), *Differential-Geometrical Methods in Statistics*,
Lecture Notes in Statistics 28, Springer,
doi:10.1007/978-1-4612-5056-2, takes the Fisher information

    g_ij(theta) = E[ d_i log p(x; theta) d_j log p(x; theta) ]

as the Riemannian metric on the parameter manifold.  The expectation
is taken here by summing (discrete support) or by the midpoint rule
(continuous support) over the supplied grid, and the score is a
central difference of log p.  For N(theta, sigma^2) the metric is
1/sigma^2 and for Bernoulli(theta) it is 1/(theta(1-theta)); both
closed forms are what the tests check.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["information_geometry"]


def information_geometry(log_p, theta, support, discrete=True, h=1e-5):
    """Fisher information matrix at theta.

    Parameters
    ----------
    log_p : callable
        log_p(x, theta) with theta a list of parameters.
    theta : array-like
        Parameter value at which the metric is evaluated.
    support : array-like
        Sample points; a grid of midpoints when discrete is False.
    discrete : bool
        True sums over the support, False applies the midpoint rule.
    h : float
        Central-difference step for the score.
    """
    th = core.vec(theta)
    d = len(th)
    if d == 0:
        raise ValueError("information_geometry: theta is empty")
    if not callable(log_p):
        raise ValueError("information_geometry: log_p must be callable")
    xs = core.vec(support)
    if len(xs) < 2:
        raise ValueError("information_geometry: support needs at least two points")
    hv = float(h)
    if hv <= 0:
        raise ValueError("information_geometry: h must be positive")
    dx = 1.0 if discrete else (xs[1] - xs[0])
    G = [[0.0] * d for _ in range(d)]
    total = 0.0
    for x in xs:
        lp = float(log_p(x, list(th)))
        w = math.exp(lp) * dx
        total += w
        sc = []
        for j in range(d):
            tp = list(th)
            tm = list(th)
            tp[j] += hv
            tm[j] -= hv
            sc.append((float(log_p(x, tp)) - float(log_p(x, tm))) / (2.0 * hv))
        for a in range(d):
            for b in range(d):
                G[a][b] += w * sc[a] * sc[b]
    return RichResult(
        title="Fisher information metric",
        summary_lines=[("parameters", d), ("support points", len(xs))],
        payload={
            "estimate": G[0][0],
            "metric": G,
            "total_mass": total,
            "n": len(xs),
            "method": "g_ij = E[d_i log p d_j log p] by quadrature over the support, Amari (1985)",
        },
    )


def cheatsheet():
    return "infgnt: Fisher information metric"
