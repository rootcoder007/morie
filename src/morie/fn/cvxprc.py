# morie.fn -- function file (rootcoder007/morie)
"""Euclidean projection -- Boyd & Vandenberghe Sec. 8.1."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_projection"]


def boyd_projection(v, C="ball", radius=1.0, lo=None, hi=None, A=None,
                    b=None):
    r"""The projection :math:`P_C(v) = \arg\min_{x \in C}
    \lVert x - v\rVert_2`.

    UNIQUE for any closed convex C, which is the property everything else
    rests on -- projected gradient, alternating projections, proximal
    methods. Onto a nonconvex set the projection can be set-valued, and
    the algorithms built on it lose their guarantees rather than merely
    slowing down.

    Projection is also NONEXPANSIVE:
    :math:`\lVert P_C(u) - P_C(v)\rVert \le \lVert u - v\rVert`. That
    is what makes projected-gradient steps safe to compose.

    Parameters
    ----------
    v : array-like
        Point to project.
    C : {"ball", "box", "simplex", "affine", "nonneg"}
        Target set.
    radius : float
        Radius, for the ball.
    lo, hi : array-like or float
        Bounds, for the box.
    A, b : array-like
        Affine set :math:`Ax = b`.

    Returns
    -------
    RichResult
        ``x``, ``distance``, ``changed``, ``set``, ``on_boundary``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    A point inside the ball is its own projection.

    >>> import numpy as np
    >>> r = boyd_projection([0.3, 0.4], "ball", radius=1.0)
    >>> bool(not r["changed"])
    True

    Outside, it is scaled to the boundary.

    >>> p = boyd_projection([3.0, 4.0], "ball", radius=1.0)
    >>> [round(float(v), 6) for v in p["x"]], round(p["distance"], 6)
    ([0.6, 0.8], 4.0)

    The simplex projection lands on the simplex exactly -- non-negative
    and summing to one, which sorting-based algorithms get subtly wrong.

    >>> s = boyd_projection([0.5, 0.7, -0.2], "simplex")["x"]
    >>> bool(abs(s.sum() - 1) < 1e-12 and np.all(s >= -1e-15))
    True

    Nonexpansiveness, the property projected-gradient methods rely on.

    >>> u, w = np.array([5.0, 0.0]), np.array([0.0, 5.0])
    >>> pu = boyd_projection(u, "ball")["x"]
    >>> pw = boyd_projection(w, "ball")["x"]
    >>> bool(np.linalg.norm(pu - pw) <= np.linalg.norm(u - w) + 1e-12)
    True
    """
    vv = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    if C == "ball":
        nrm = float(np.linalg.norm(vv))
        x = vv if nrm <= radius else vv * (radius / nrm)
        boundary = bool(abs(float(np.linalg.norm(x)) - radius) < 1e-12)
    elif C == "box":
        loa = -np.inf if lo is None else np.asarray(lo, dtype=float)
        hia = np.inf if hi is None else np.asarray(hi, dtype=float)
        x = np.clip(vv, loa, hia)
        boundary = bool(np.any(x != vv))
    elif C == "nonneg":
        x = np.maximum(vv, 0.0)
        boundary = bool(np.any(vv < 0))
    elif C == "simplex":
        # Duchi et al.'s sorting algorithm: exact, not iterative.
        u = np.sort(vv)[::-1]
        css = np.cumsum(u) - 1.0
        idx = np.arange(1, vv.size + 1)
        cond = u - css / idx > 0
        rho = int(idx[cond][-1])
        theta = css[rho - 1] / rho
        x = np.maximum(vv - theta, 0.0)
        boundary = bool(np.any(x == 0))
    elif C == "affine":
        Am = np.atleast_2d(np.asarray(A, dtype=float))
        bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
        # v + A'(AA')^{-1}(b - Av)
        x = vv + Am.T @ np.linalg.lstsq(Am @ Am.T, bv - Am @ vv,
                                        rcond=None)[0]
        boundary = True
    else:
        raise ValueError(
            'C must be one of "ball", "box", "simplex", "affine", "nonneg"')
    dist = float(np.linalg.norm(x - vv))
    return RichResult(
        title=f"Projection onto {C}",
        summary_lines=[("set", C), ("distance", dist),
                       ("changed", bool(dist > 1e-15))],
        payload={
            "x": x, "distance": dist, "changed": bool(dist > 1e-15),
            "set": C, "on_boundary": boundary,
            "method": "boyd_projection",
        },
    )


def cheatsheet():
    return "cvxprc: unique and NONEXPANSIVE for convex C -- both fail on a nonconvex set"
