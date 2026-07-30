# morie.fn -- function file (rootcoder007/morie)
"""Hinge loss -- Boyd & Vandenberghe Sec. 6.1 / 8.6."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_hinge_loss"]


def boyd_hinge_loss(u, margin=1.0):
    r"""The hinge :math:`\phi(u) = \max(0, 1 - u)`.

    Convex, piecewise linear, and NOT differentiable at the kink
    :math:`u = 1` -- which is the whole reason SVM training uses
    subgradient or dual methods rather than plain Newton.

    Its flat region is what makes it a margin loss: a point classified
    correctly and beyond the margin contributes exactly zero, so it exerts
    no pull at all on the boundary. That is the property the support-vector
    story rests on -- only the points at or inside the margin have nonzero
    subgradient, and only they determine the solution.

    Parameters
    ----------
    u : array-like
        Margin values, typically :math:`y_i(w^\top x_i + b)`.
    margin : float
        The kink location; 1 is the standard choice.

    Returns
    -------
    RichResult
        ``loss`` (elementwise), ``total``, ``mean``, ``subgradient``,
        ``active`` (points with nonzero loss), ``n_support``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Correct and beyond the margin costs nothing; inside the margin costs
    the shortfall.

    >>> import numpy as np
    >>> r = boyd_hinge_loss([2.0, 1.0, 0.5, -1.0])
    >>> [float(v) for v in r["loss"]]
    [0.0, 0.0, 0.5, 2.0]

    Only the points at or inside the margin carry a subgradient -- these
    are the support vectors.

    >>> int(r["n_support"])
    2

    The kink is real: the subgradient jumps at u = 1 rather than passing
    smoothly through zero.

    >>> (float(boyd_hinge_loss([0.999])["subgradient"][0]),
    ...  float(boyd_hinge_loss([1.001])["subgradient"][0]))
    (-1.0, 0.0)
    """
    uv = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    m = float(margin)
    loss = np.maximum(0.0, m - uv)
    # Subgradient: -1 strictly inside the margin, 0 strictly outside. At the
    # kink any value in [-1, 0] is a valid subgradient; 0 is the convention.
    sub = np.where(uv < m, -1.0, 0.0)
    active = loss > 0
    return RichResult(
        title="Hinge loss",
        summary_lines=[("n", int(uv.size)), ("total", float(loss.sum())),
                       ("mean", float(loss.mean())),
                       ("support vectors", int(active.sum()))],
        payload={
            "loss": loss, "total": float(loss.sum()),
            "mean": float(loss.mean()), "subgradient": sub,
            "active": active, "n_support": int(active.sum()),
            "margin": m, "differentiable": bool(not np.any(uv == m)),
            "method": "boyd_hinge_loss",
        },
    )


def cheatsheet():
    return "cvxhng: flat past the margin, so only margin-violating points move the boundary; kink at u=1"
