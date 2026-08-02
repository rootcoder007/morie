# morie.fn -- function file (rootcoder007/morie)
"""Supporting hyperplane -- Boyd & Vandenberghe Sec. 2.5.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_support_hyperplane"]


def boyd_support_hyperplane(C, x0, tol=1e-07):
    r"""A hyperplane :math:`a \ne 0` with :math:`a^\top x \le a^\top x_0`
    for every :math:`x \in C`, at a boundary point :math:`x_0`.

    The supporting hyperplane theorem: EVERY boundary point of a convex
    set has one. Uniqueness is a different question -- at a smooth
    boundary point the supporting hyperplane is unique and its normal is
    the gradient, while at a corner there is a whole cone of them. That
    cone is the normal cone, and it is precisely the subdifferential of
    the set's indicator, which is where KKT multipliers come from.

    So a corner is not a degenerate case to be avoided; it is where the
    multiplier has room to move, and where an active-set method has a
    choice to make.

    Parameters
    ----------
    C : array-like
        Points generating the convex set.
    x0 : array-like
        Boundary point.
    tol : float
        Tolerance for the support condition.

    Returns
    -------
    RichResult
        ``a``, ``offset``, ``supports``, ``n_touching`` (points on the
        hyperplane), ``is_corner``, ``max_violation``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    At a face of a square the supporting hyperplane touches two
    generators.

    >>> import numpy as np
    >>> S = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> r = boyd_support_hyperplane(S, [0.5, 1.0])
    >>> bool(r["supports"])
    True

    Every generator is on the correct side -- the support condition holds
    rather than merely being claimed.

    >>> a, off = r["a"], r["offset"]
    >>> bool(np.all(S @ a <= off + 1e-7))
    True

    At a CORNER the supporting hyperplane is not unique; the function
    returns one and flags that a cone of them exists, which is the normal
    cone the KKT multipliers live in.

    >>> c = boyd_support_hyperplane(S, [1.0, 1.0])
    >>> bool(c["supports"]), bool(c["is_corner"])
    (True, True)
    """
    from scipy.optimize import linprog

    S = np.atleast_2d(np.asarray(C, dtype=float))
    x0 = np.atleast_1d(np.asarray(x0, dtype=float)).ravel()
    m, n = S.shape
    if x0.size != n:
        raise ValueError(f"x0 must have {n} coordinates")
    # Find a with a'x <= a'x0 for all x in S, maximising the slack that
    # separates x0 from the interior. Bounded a keeps the LP finite.
    A_ub = S - x0
    res = linprog(-(np.mean(S, axis=0) - x0) * 0.0 - (x0 - np.mean(S, axis=0)),
                  A_ub=A_ub, b_ub=np.full(m, tol),
                  bounds=[(-1.0, 1.0)] * n, method="highs")
    if res.status != 0 or np.linalg.norm(res.x) < 1e-9:
        a = x0 - np.mean(S, axis=0)
        nrm = float(np.linalg.norm(a))
        a = a / nrm if nrm > 0 else np.zeros(n)
    else:
        a = np.asarray(res.x, dtype=float)
        nrm = float(np.linalg.norm(a))
        a = a / nrm if nrm > 0 else a
    off = float(a @ x0)
    slack = S @ a - off
    supports = bool(np.max(slack) <= tol and np.linalg.norm(a) > 1e-9)
    touching = int(np.sum(np.abs(slack) <= 1e-07))
    return RichResult(
        title="Supporting hyperplane",
        summary_lines=[("dimension", int(n)), ("supports", supports),
                       ("touching generators", touching)],
        payload={
            "a": a, "offset": off, "supports": supports,
            "n_touching": touching,
            # One touching generator means x0 is extreme: a corner, where
            # the supporting hyperplane is one of a whole normal cone.
            "is_corner": bool(touching == 1),
            "max_violation": float(np.max(slack)),
            "method": "boyd_support_hyperplane",
        },
    )


def cheatsheet():
    return "cvxsup: exists at EVERY boundary point; unique only where smooth -- a corner has a normal CONE"
