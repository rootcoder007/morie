# morie.fn -- function file (rootcoder007/morie)
"""Subgradient -- Boyd & Vandenberghe Sec. 3.1.2 / 9.4."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_subgradient"]


def boyd_subgradient(f, x, g, y_samples=None, n_probe=64, radius=1.0,
                     seed=0):
    r"""Verify that g is a subgradient of f at x:

    .. math::
        f(y) \ge f(x) + g^\top (y - x) \quad \text{for all } y.

    The definition is a GLOBAL statement, which is why it exists only for
    convex functions -- for a nonconvex f the affine minorant fails
    somewhere however g is chosen. At a differentiable point the
    subdifferential is the single gradient; at a kink it is an interval,
    and any of its elements is a valid subgradient.

    Verification here is by sampling, so it can REFUTE the claim but never
    establish it. A pass means no counterexample was found among the
    probes, which is reported as such rather than as proof.

    Parameters
    ----------
    f : callable
        The function.
    x : array-like
        Point.
    g : array-like
        Candidate subgradient.
    y_samples : array-like, optional
        Explicit points at which to test. Random probes are used when
        omitted.
    n_probe, radius, seed
        Controls for the random probes.

    Returns
    -------
    RichResult
        ``is_subgradient`` (no counterexample found), ``violations``,
        ``worst_gap``, ``n_tested``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    At the kink of the absolute value any slope in [-1, 1] is a
    subgradient.

    >>> boyd_subgradient(abs, 0.0, 0.5)["is_subgradient"]
    True
    >>> boyd_subgradient(abs, 0.0, 0.0)["is_subgradient"]
    True

    Outside that interval it is not, and a counterexample is produced.

    >>> r = boyd_subgradient(abs, 0.0, 1.5)
    >>> bool(r["is_subgradient"]), int(r["violations"] > 0)
    (False, 1)

    Away from the kink the subdifferential collapses to the derivative.

    >>> boyd_subgradient(abs, 2.0, 1.0)["is_subgradient"]
    True
    >>> boyd_subgradient(abs, 2.0, 0.5)["is_subgradient"]
    False
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    gv = np.atleast_1d(np.asarray(g, dtype=float)).ravel()
    if gv.size != xv.size:
        raise ValueError(f"g has {gv.size} entries but x has {xv.size}")
    if y_samples is None:
        rng = np.random.default_rng(seed)
        ys = xv + rng.uniform(-radius, radius, (int(n_probe), xv.size))
        # Include the axis directions explicitly: a wrong slope shows up
        # most sharply straight along a coordinate, and random probes in
        # several dimensions can miss it.
        eye = np.eye(xv.size)
        ys = np.vstack([ys, xv + radius * eye, xv - radius * eye])
    else:
        ys = np.atleast_2d(np.asarray(y_samples, dtype=float))
        if ys.shape[1] != xv.size:
            ys = ys.T
    fx = float(f(xv[0] if xv.size == 1 else xv))
    gaps = np.empty(ys.shape[0])
    for i, y in enumerate(ys):
        fy = float(f(y[0] if xv.size == 1 else y))
        gaps[i] = fy - (fx + float(gv @ (y - xv)))
    viol = int(np.sum(gaps < -1e-09))
    return RichResult(
        title="Subgradient check",
        summary_lines=[("probes", int(ys.shape[0])), ("violations", viol),
                       ("worst gap", float(gaps.min()))],
        warnings=["this samples the inequality: it can refute the "
                  "subgradient claim but never establish it"],
        payload={
            "is_subgradient": bool(viol == 0), "violations": viol,
            "worst_gap": float(gaps.min()), "n_tested": int(ys.shape[0]),
            "gaps": gaps, "f_at_x": fx,
            "method": "boyd_subgradient",
        },
    )


def cheatsheet():
    return "cvxsbp: a GLOBAL affine minorant, so it exists only for convex f; sampling refutes, never proves"
