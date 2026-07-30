# morie.fn -- function file (rootcoder007/morie)
"""Indicator function -- Boyd & Vandenberghe Sec. 3.1.2."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .cvxprc import boyd_projection

__all__ = ["boyd_indicator"]


def boyd_indicator(x, C="ball", tol=1e-09, **set_kw):
    r"""The convex indicator

    .. math::
        I_C(x) = \begin{cases} 0 & x \in C \\
                               +\infty & x \notin C.\end{cases}

    NOT the 0/1 indicator of probability, and the difference is the whole
    construction: the value :math:`+\infty` is what turns a constrained
    problem :math:`\min f(x)` over C into the unconstrained
    :math:`\min f(x) + I_C(x)`. A 0/1 indicator would merely add a
    bounded penalty and change the answer.

    Its subdifferential at a boundary point is the NORMAL CONE, which is
    where the KKT multipliers come from, and its proximal operator is
    exactly the projection onto C -- which is how projected gradient turns
    out to be a special case of the proximal method.

    Parameters
    ----------
    x : array-like
        Point to test.
    C : str
        Set, as for :func:`~morie.fn.cvxprc.boyd_projection`.
    tol : float
        Membership tolerance.
    **set_kw
        Set parameters.

    Returns
    -------
    RichResult
        ``value`` (0 or inf), ``in_set``, ``distance``, ``prox`` (the
        projection), ``on_boundary``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Zero inside, infinite outside.

    >>> boyd_indicator([0.3, 0.4], "ball")["value"]
    0.0
    >>> boyd_indicator([3.0, 4.0], "ball")["value"]
    inf

    The proximal operator of the indicator IS the projection, which is
    why projected gradient is a proximal method.

    >>> import numpy as np
    >>> p = boyd_indicator([3.0, 4.0], "ball")["prox"]
    >>> [round(float(v), 6) for v in p]
    [0.6, 0.8]

    A finite penalty would not enforce the constraint; the infinity is
    the mechanism, not a formality.

    >>> bool(np.isinf(boyd_indicator([2.0], "ball")["value"]))
    True
    """
    proj = boyd_projection(x, C, **set_kw)
    dist = float(proj["distance"])
    inside = bool(dist <= tol)
    return RichResult(
        title=f"Indicator of {C}",
        summary_lines=[("in set", inside), ("distance", dist),
                       ("value", 0.0 if inside else float("inf"))],
        payload={
            "value": 0.0 if inside else float("inf"), "in_set": inside,
            "distance": dist, "prox": proj["x"],
            "on_boundary": bool(proj["on_boundary"] and inside),
            "set": C, "method": "boyd_indicator",
        },
    )


def cheatsheet():
    return "cvxind: +inf (not 1) is what enforces the constraint; its prox IS the projection"
