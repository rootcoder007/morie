# morie.fn -- function file (rootcoder007/morie)
"""Steepest descent direction -- Boyd & Vandenberghe Sec. 9.4."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_steepest_desc"]


def boyd_steepest_desc(grad, norm=2, P=None):
    r"""The normalised steepest-descent direction

    .. math::
        \Delta x = \arg\min\{\nabla f(x)^\top v : \lVert v\rVert \le 1\}.

    "Steepest" is relative to a NORM, and different norms give genuinely
    different directions -- the Euclidean answer is the negative gradient,
    but the :math:`\ell_1` answer is a single coordinate step and the
    :math:`\ell_\infty` answer is a sign vector. Calling the negative
    gradient "the" steepest direction quietly assumes a norm.

    The quadratic norm :math:`\lVert v\rVert_P = (v^\top Pv)^{1/2}` gives
    :math:`-P^{-1}\nabla f`, so Newton's method IS steepest descent in the
    norm defined by the Hessian. That is the cleanest way to see why
    Newton is affine invariant while gradient descent is not.

    Parameters
    ----------
    grad : array-like
        Gradient.
    norm : {1, 2, "inf"} or "quadratic"
        Norm defining steepness.
    P : array-like, optional
        Positive definite matrix, for the quadratic norm.

    Returns
    -------
    RichResult
        ``direction`` (unit in the chosen norm), ``descent_rate``
        (:math:`\nabla f^\top \Delta x`), ``dual_norm``, ``is_descent``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    In the Euclidean norm it is the normalised negative gradient.

    >>> import numpy as np
    >>> r = boyd_steepest_desc([3.0, 4.0], norm=2)
    >>> [round(float(v), 4) for v in r["direction"]]
    [-0.6, -0.8]

    In the l1 norm it is a single COORDINATE step -- a different
    direction entirely, which is why "steepest" needs a norm to mean
    anything.

    >>> [float(v) for v in boyd_steepest_desc([3.0, 4.0], norm=1)["direction"]]
    [0.0, -1.0]

    In the sup norm it is a sign vector, so every coordinate moves.

    >>> [float(v) for v in boyd_steepest_desc([3.0, -4.0], norm="inf")["direction"]]
    [-1.0, 1.0]

    In the norm defined by a positive definite P the direction is
    -P^-1 g, which is exactly the Newton step when P is the Hessian.

    >>> P = np.array([[4.0, 0.0], [0.0, 1.0]])
    >>> d = boyd_steepest_desc([4.0, 1.0], norm="quadratic", P=P)["direction"]
    >>> bool(np.allclose(d / np.linalg.norm(d),
    ...                  -np.linalg.solve(P, [4.0, 1.0])
    ...                  / np.linalg.norm(np.linalg.solve(P, [4.0, 1.0]))))
    True

    Every version is a genuine descent direction.

    >>> all(boyd_steepest_desc([3.0, 4.0], n)["is_descent"]
    ...     for n in (1, 2, "inf"))
    True
    """
    g = np.atleast_1d(np.asarray(grad, dtype=float)).ravel()
    if g.size == 0:
        raise ValueError("grad must be non-empty")
    if norm == "quadratic":
        if P is None:
            raise ValueError("P is required for the quadratic norm")
        Pm = np.atleast_2d(np.asarray(P, dtype=float))
        w = np.linalg.eigvalsh(0.5 * (Pm + Pm.T))
        if w.min() <= 0:
            raise ValueError("P must be positive definite")
        raw = -np.linalg.solve(Pm, g)
        scale = float(np.sqrt(max(raw @ Pm @ raw, 1e-300)))
        d = raw / scale
        dn = float(np.sqrt(g @ np.linalg.solve(Pm, g)))
    elif norm == 2:
        nrm = float(np.linalg.norm(g))
        d = -g / nrm if nrm > 0 else np.zeros_like(g)
        dn = nrm
    elif norm == 1:
        # The minimiser over the l1 ball sits at a vertex: one coordinate.
        i = int(np.argmax(np.abs(g)))
        d = np.zeros_like(g)
        d[i] = -np.sign(g[i])
        dn = float(np.max(np.abs(g)))
    elif norm in ("inf", np.inf, float("inf")):
        d = -np.sign(g)
        dn = float(np.sum(np.abs(g)))
    else:
        raise ValueError('norm must be 1, 2, "inf" or "quadratic"')
    rate = float(g @ d)
    return RichResult(
        title=f"Steepest descent (l{norm})",
        summary_lines=[("norm", str(norm)), ("descent rate", rate),
                       ("dual norm", dn)],
        payload={
            "direction": d, "descent_rate": rate, "dual_norm": dn,
            "is_descent": bool(rate < 0), "norm": str(norm),
            "method": "boyd_steepest_desc",
        },
    )


def cheatsheet():
    return "cvxstd: 'steepest' needs a NORM; Newton IS steepest descent in the Hessian norm"
