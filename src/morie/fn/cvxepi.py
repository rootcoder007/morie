# morie.fn -- function file (rootcoder007/morie)
"""Epigraph -- Boyd & Vandenberghe Sec. 3.1.7."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_epigraph"]


def boyd_epigraph(f, x, t):
    r"""Membership in :math:`\operatorname{epi} f = \{(x, t) : f(x) \le t\}`.

    The epigraph is the bridge between convex FUNCTIONS and convex SETS: a
    function is convex exactly when its epigraph is a convex set, so every
    theorem about convex sets becomes a theorem about convex functions for
    free.

    It is also the standard modelling trick. An objective
    :math:`\min f(x)` becomes :math:`\min t` subject to
    :math:`f(x) \le t`, which is what lets a solver that only handles
    linear objectives handle any convex one.

    Parameters
    ----------
    f : callable
        The function.
    x : array-like
        Point(s) at which to test.
    t : array-like
        Candidate epigraph height(s).

    Returns
    -------
    RichResult
        ``in_epigraph``, ``fx``, ``slack`` (:math:`t - f(x)`),
        ``on_boundary``, ``fraction_inside``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    A point above the graph is in the epigraph; below is not.

    >>> r = boyd_epigraph(lambda x: x ** 2, [1.0, 1.0], [2.0, 0.5])
    >>> [bool(v) for v in r["in_epigraph"]]
    [True, False]

    Slack is the vertical distance to the graph, and is zero exactly on
    the boundary.

    >>> b = boyd_epigraph(lambda x: x ** 2, [2.0], [4.0])
    >>> float(b["slack"][0]), bool(b["on_boundary"][0])
    (0.0, True)

    Convexity of the function shows up as convexity of the epigraph: the
    midpoint of two epigraph points is itself in the epigraph.

    >>> import numpy as np
    >>> f = lambda x: x ** 2
    >>> p, q = (-2.0, 5.0), (3.0, 10.0)
    >>> mid_x, mid_t = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    >>> bool(boyd_epigraph(f, [mid_x], [mid_t])["in_epigraph"][0])
    True
    """
    if not callable(f):
        raise ValueError("f must be callable")
    xv = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    tv = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    if xv.size != tv.size:
        raise ValueError(f"x has {xv.size} entries but t has {tv.size}")
    fx = np.asarray([float(f(v)) for v in xv], dtype=float)
    slack = tv - fx
    inside = slack >= 0
    return RichResult(
        title="Epigraph membership",
        summary_lines=[("points", int(xv.size)),
                       ("inside", int(inside.sum())),
                       ("on boundary", int(np.sum(np.abs(slack) <= 1e-12)))],
        payload={
            "in_epigraph": inside, "fx": fx, "slack": slack,
            "on_boundary": np.abs(slack) <= 1e-12,
            "fraction_inside": float(inside.mean()),
            "method": "boyd_epigraph",
        },
    )


def cheatsheet():
    return "cvxepi: f convex IFF epi f convex; min f(x) becomes min t s.t. f(x) <= t"
