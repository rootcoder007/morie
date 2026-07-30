# morie.fn -- function file (rootcoder007/morie)
"""Generalized inequalities -- Boyd & Vandenberghe Sec. 2.4."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_generalized_p"]


def boyd_generalized_p(x, y, K="nonneg", tol=1e-09):
    r"""The generalized inequality :math:`x \preceq_K y \iff y - x \in K`
    for a proper cone K.

    The essential difference from the scalar case is that
    :math:`\preceq_K` is a PARTIAL order, not a total one: two points can
    be incomparable, with neither :math:`x \preceq_K y` nor
    :math:`y \preceq_K x`. That is not a defect of the definition, it is
    why vector-valued optimisation has a Pareto FRONTIER rather than a
    single optimum, and why "the minimum" has to be replaced by "minimal
    elements".

    Supported cones: the non-negative orthant (componentwise ordering),
    the second-order cone, and the positive semi-definite cone (the
    Loewner order).

    Parameters
    ----------
    x, y : array-like
        Points, or symmetric matrices for the PSD cone.
    K : {"nonneg", "soc", "psd"}
        The cone.
    tol : float
        Membership tolerance.

    Returns
    -------
    RichResult
        ``precedes``, ``succeeds``, ``comparable``, ``strict``,
        ``margin``, ``cone``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Componentwise ordering on the non-negative orthant.

    >>> r = boyd_generalized_p([1.0, 2.0], [3.0, 5.0])
    >>> bool(r["precedes"]), bool(r["strict"])
    (True, True)

    Two points can be INCOMPARABLE -- neither precedes the other. This is
    the fact that creates Pareto frontiers.

    >>> i = boyd_generalized_p([1.0, 5.0], [3.0, 2.0])
    >>> bool(i["precedes"]), bool(i["succeeds"]), bool(i["comparable"])
    (False, False, False)

    The Loewner order on the PSD cone: B - A must be positive
    semi-definite, which is strictly stronger than every entry being
    larger.

    >>> import numpy as np
    >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> B = np.array([[2.0, 0.0], [0.0, 3.0]])
    >>> bool(boyd_generalized_p(A, B, "psd")["precedes"])
    True

    Entrywise larger does NOT imply Loewner larger -- the trap this
    ordering exists to make visible.

    >>> C = np.array([[2.0, 2.5], [2.5, 3.0]])
    >>> bool(np.all(C >= A)), bool(boyd_generalized_p(A, C, "psd")["precedes"])
    (True, False)
    """
    if K == "psd":
        Xm = np.atleast_2d(np.asarray(x, dtype=float))
        Ym = np.atleast_2d(np.asarray(y, dtype=float))
        D = 0.5 * ((Ym - Xm) + (Ym - Xm).T)
        w = np.linalg.eigvalsh(D)
        prec = bool(w.min() >= -tol)
        w2 = np.linalg.eigvalsh(-D)
        succ = bool(w2.min() >= -tol)
        margin = float(w.min())
        strict = bool(w.min() > tol)
    elif K == "soc":
        xv = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        d = yv - xv
        # (u, t) in the SOC iff ||u||_2 <= t, with t the LAST coordinate.
        prec = bool(np.linalg.norm(d[:-1]) <= d[-1] + tol)
        e = xv - yv
        succ = bool(np.linalg.norm(e[:-1]) <= e[-1] + tol)
        margin = float(d[-1] - np.linalg.norm(d[:-1]))
        strict = bool(margin > tol)
    elif K == "nonneg":
        xv = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        if xv.size != yv.size:
            raise ValueError(f"x has {xv.size} entries but y has {yv.size}")
        d = yv - xv
        prec = bool(np.all(d >= -tol))
        succ = bool(np.all(d <= tol))
        margin = float(d.min())
        strict = bool(np.all(d > tol))
    else:
        raise ValueError('K must be "nonneg", "soc" or "psd"')
    return RichResult(
        title=f"Generalized inequality ({K})",
        summary_lines=[("cone", K), ("x <= y", prec), ("y <= x", succ),
                       ("comparable", bool(prec or succ))],
        payload={
            "precedes": prec, "succeeds": succ,
            "comparable": bool(prec or succ), "strict": strict,
            "margin": margin, "cone": K,
            "partial_order_note": "incomparable pairs are why vector "
                                  "problems have a Pareto frontier rather "
                                  "than a single optimum",
            "method": "boyd_generalized_p",
        },
    )


def cheatsheet():
    return "cvxgsa: a PARTIAL order -- incomparable pairs are exactly why Pareto frontiers exist"
