# morie.fn -- function file (rootcoder007/morie)
"""Minimum volume covering ellipsoid -- Boyd & Vandenberghe Sec. 8.4.1."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_minvol_ellipsoid"]


def boyd_minvol_ellipsoid(X, tol=1e-07, max_iter=10000):
    r"""Löwner-John ellipsoid: the smallest ellipsoid covering the points.

    Minimise :math:`\log\det B^{-1}` subject to
    :math:`\lVert Bx_i + d\rVert_2 \le 1` -- convex in :math:`(B, d)`
    even though the volume itself is not, which is the reformulation
    that makes the problem tractable at all.

    Solved by Khachiyan's algorithm on the DUAL, which reads as a
    weighting problem: find weights on the points, summing to one, whose
    weighted second-moment matrix has maximum determinant. Its solution
    is supported on at most :math:`d(d+3)/2` points, so the ellipsoid is
    pinned by a handful of extreme observations and the interior of the
    cloud is irrelevant to it. That is exactly what makes it a good
    outlier detector and a terrible summary of typical values.

    John's theorem gives the guarantee that makes it useful: shrinking
    the ellipsoid by a factor of :math:`d` about its centre leaves an
    ellipsoid CONTAINED in the convex hull, so one ellipsoid brackets
    the hull from both sides within a dimension-dependent factor.

    Parameters
    ----------
    X : array-like
        Points, ``(n, d)``, one row per point.
    tol : float
        Khachiyan convergence tolerance on the excess of the worst
        point's Mahalanobis value over ``d + 1``.
    max_iter : int
        Iteration cap.

    Returns
    -------
    RichResult
        ``center``, ``A`` (shape matrix, ``(x-c)'A(x-c) <= 1``), ``B``
        (its matrix square root), ``axes``, ``volume``, ``weights``,
        ``support``, ``mahalanobis``, ``covers_all``, ``converged``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.
    Khachiyan, L. G. (1996). Rounding of polytopes in the real number
        model of computation. *Mathematics of Operations Research*,
        21(2), 307-320.

    Examples
    --------
    The corners of the unit square. By symmetry no ellipse can beat the
    circumscribed CIRCLE, centred at (1/2, 1/2) with radius
    sqrt(2)/2 -- and all four corners lie exactly on it.

    >>> import numpy as np
    >>> sq = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> r = boyd_minvol_ellipsoid(sq)
    >>> [round(float(v), 5) for v in r["center"]]
    [0.5, 0.5]
    >>> [round(float(v), 5) for v in r["axes"]]
    [0.70711, 0.70711]
    >>> bool(np.all(np.abs(r["mahalanobis"] - 1.0) < 1e-05))
    True

    Its area is pi r^2 = pi/2.

    >>> round(float(r["volume"]), 5)
    1.5708

    Adding a point INSIDE the hull changes nothing -- the solution is
    supported on extreme points only, and the interior weight is zero.

    >>> inner = np.vstack([sq, [[0.5, 0.5]]])
    >>> r2 = boyd_minvol_ellipsoid(inner)
    >>> bool(abs(r2["volume"] - r["volume"]) < 1e-05)
    True
    >>> bool(r2["weights"][4] < 1e-06)
    True

    Every point is covered, by construction.

    >>> bool(r2["covers_all"])
    True

    Two points in the plane cannot pin down an ellipse -- the problem is
    unbounded below and refusing is the only honest answer.

    >>> boyd_minvol_ellipsoid([[0.0, 0.0], [1.0, 1.0]])
    Traceback (most recent call last):
        ...
    ValueError: need at least 3 points in 2 dimensions, got 2
    """
    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    if Xm.ndim != 2:
        raise ValueError("X must be a 2-d array of points")
    if not np.all(np.isfinite(Xm)):
        raise ValueError("X contains non-finite entries")
    n, d = Xm.shape
    # d+1 affinely independent points are the minimum that can bound an
    # ellipsoid at all; the ALSO-true fact that the optimal weights are
    # supported on at most d(d+3)/2 points is an upper bound on the
    # support, not a requirement on the input.
    need = d + 1
    if n < need:
        raise ValueError(
            f"need at least {need} points in {d} dimensions, got {n}")
    # Khachiyan works on the lifted points [x; 1]: the lift turns the
    # centre into part of the shape matrix, so a single determinant
    # maximisation handles both.
    Q = np.hstack([Xm, np.ones((n, 1))]).T
    u = np.full(n, 1.0 / n)
    converged = False
    for _ in range(int(max_iter)):
        M = Q @ np.diag(u) @ Q.T
        try:
            g = np.einsum("ij,ji->i", Q.T, np.linalg.solve(M, Q))
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "points are degenerate: they lie in a lower-dimensional "
                "affine subspace, so no covering ellipsoid has finite "
                "volume") from exc
        up = int(np.argmax(g))
        pos = np.flatnonzero(u > 0.0)
        down = int(pos[np.argmin(g[pos])])
        excess = g[up] - (d + 1.0)
        deficit = (d + 1.0) - g[down]
        if max(excess, deficit) <= tol * (d + 1):
            converged = True
            break
        # Wolfe-Atwood: plain Khachiyan only ever ADDS weight, so an
        # interior point keeps a shrinking-but-nonzero share forever and
        # the support never becomes exact. The away step removes weight
        # from the worst-served supported point and, capped, drives it
        # to exactly zero.
        j = up if excess >= deficit else down
        if abs(g[j] - 1.0) < 1e-12:
            # g == 1 means the point sits at the current centre: the
            # unconstrained away step is infinite, so take the capped
            # one directly rather than dividing by zero to get there.
            step = -u[j] / (1.0 - u[j])
        else:
            step = (g[j] - d - 1.0) / ((d + 1.0) * (g[j] - 1.0))
        if step < 0.0:
            step = max(step, -u[j] / (1.0 - u[j]))
        u = (1.0 - step) * u
        u[j] += step
        u[u < 0.0] = 0.0
    c = u @ Xm
    # A = (1/d) (sum u_i x_i x_i' - c c')^-1 is the shape matrix of
    # {x : (x-c)'A(x-c) <= 1}.
    S = (Xm * u[:, None]).T @ Xm - np.outer(c, c)
    S = 0.5 * (S + S.T)
    A = np.linalg.inv(S) / d
    A = 0.5 * (A + A.T)
    evals, evecs = np.linalg.eigh(A)
    if evals[0] <= 0:
        raise ValueError("degenerate point set: shape matrix is singular")
    axes = 1.0 / np.sqrt(evals)
    from math import gamma, pi

    unit_vol = pi ** (d / 2.0) / gamma(d / 2.0 + 1.0)
    diff = Xm - c
    maha = np.einsum("ij,jk,ik->i", diff, A, diff)
    return RichResult(
        title="Minimum volume covering ellipsoid",
        summary_lines=[("n", int(n)), ("d", int(d)),
                       ("volume", float(unit_vol * np.prod(axes))),
                       ("support", int(np.sum(u > 1e-06))),
                       ("converged", bool(converged))],
        payload={
            "center": c, "A": A,
            "B": evecs @ np.diag(np.sqrt(evals)) @ evecs.T,
            "axes": axes[::-1].copy(),
            "volume": float(unit_vol * np.prod(axes)),
            "weights": u, "support": np.flatnonzero(u > 1e-06),
            "mahalanobis": maha,
            "covers_all": bool(np.max(maha) <= 1.0 + 1e-06),
            "converged": bool(converged),
            "method": "boyd_minvol_ellipsoid",
        },
    )


def cheatsheet():
    return "cvxell: pinned by a handful of EXTREME points -- good outlier detector, bad summary of typical values"
