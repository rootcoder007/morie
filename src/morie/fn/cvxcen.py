# morie.fn -- function file (rootcoder007/morie)
"""Central path of the barrier -- Boyd & Vandenberghe Sec. 11.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from .cvxipm import _centering

__all__ = ["boyd_central_path"]


def boyd_central_path(f0, f, t, x0=None):
    r"""Trace :math:`x^{\star}(t) = \arg\min\ t f_0(x) + \phi(x)`.

    Where :func:`boyd_interior_point` follows the path only far enough
    to answer the problem, this returns the path itself -- the curve the
    barrier method actually walks, parameterised by :math:`t`.

    Its two endpoints are the interesting ones. As :math:`t \to 0` the
    objective disappears and :math:`x^{\star}(t)` tends to the ANALYTIC
    CENTRE of the feasible set, a point that depends on how the
    constraints were WRITTEN, not just on the set they describe:
    duplicating an inequality moves it. As :math:`t \to \infty` the
    barrier's pull vanishes and the path converges to the optimum, from
    strictly inside.

    Every point on the path is optimal for something -- it satisfies the
    KKT conditions of the original problem with the complementary
    slackness condition RELAXED from :math:`\lambda_i f_i(x) = 0` to
    :math:`\lambda_i f_i(x) = -1/t`. That is why the path is central
    rather than arbitrary, and why the duality gap along it is exactly
    :math:`m/t`.

    Parameters
    ----------
    f0 : callable
        Objective.
    f : sequence of callable
        Inequality constraints ``f_i(x) <= 0``.
    t : array-like
        Barrier parameters to solve at. Sorted ascending internally, so
        each solve warm-starts from the previous point.
    x0 : array-like
        Strictly feasible start.

    Returns
    -------
    RichResult
        ``t`` (sorted), ``path`` (one row per t), ``objective``,
        ``gap`` (``m/t``), ``suboptimality_bounded``, ``slack``
        (the constraint values along the path), ``strictly_feasible``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Minimise ``x^2/2`` subject to ``x >= 1``. Centering gives
    ``t*x = 1/(x-1)``, so ``x*(t) = (1 + sqrt(1 + 4/t))/2`` in closed
    form and the whole path can be checked against it.

    >>> import numpy as np
    >>> obj = lambda x: 0.5 * x[0] ** 2
    >>> con = [lambda x: 1.0 - x[0]]
    >>> r = boyd_central_path(obj, con, [1.0, 10.0, 100.0], x0=[2.0])
    >>> [round(float(v), 6) for v in r["path"][:, 0]]
    [1.618034, 1.091608, 1.009902]
    >>> [round(float((1 + np.sqrt(1 + 4 / t)) / 2), 6)
    ...  for t in (1.0, 10.0, 100.0)]
    [1.618034, 1.091608, 1.009902]

    The gap along the path is exactly m/t, and it BOUNDS the true
    suboptimality at every point -- that is the property that makes the
    path worth following rather than merely converging.

    >>> [round(float(g), 6) for g in r["gap"]]
    [1.0, 0.1, 0.01]
    >>> bool(np.all(r["objective"] - 0.5 <= r["gap"] + 1e-09))
    True

    The path approaches the optimum from strictly INSIDE: the objective
    falls monotonically toward 1/2 without ever reaching it, and the
    constraint is never satisfied with equality.

    >>> bool(np.all(np.diff(r["objective"]) < 0))
    True
    >>> bool(np.all(r["slack"] < 0))
    True

    Small t pulls toward the analytic centre instead of the optimum.
    With a second constraint ``x <= 3`` the centre of ``[1, 3]`` is 2,
    and the path at tiny t sits near it rather than near the answer.

    >>> two = boyd_central_path(obj, [con[0], lambda x: x[0] - 3.0],
    ...                         [1e-06, 1e+04], x0=[2.0])
    >>> round(float(two["path"][0, 0]), 4)
    2.0
    >>> round(float(two["path"][1, 0]), 4)
    1.0001
    """
    if not callable(f0):
        raise TypeError("f0 must be callable")
    fs = list(f)
    for k, fi in enumerate(fs):
        if not callable(fi):
            raise TypeError(f"f[{k}] must be callable")
    if not fs:
        raise ValueError("no constraints: there is no barrier and hence "
                         "no central path")
    tv = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    if tv.size == 0:
        raise ValueError("t is empty")
    if np.any(tv <= 0):
        raise ValueError("every t must be positive")
    if x0 is None:
        raise ValueError("x0 is required and must be strictly feasible")
    x = np.atleast_1d(np.asarray(x0, dtype=float)).ravel().copy()
    for k, fi in enumerate(fs):
        v = float(fi(x))
        if not (v < 0):
            raise ValueError(
                f"x0 is not strictly feasible: constraint {k} has "
                f"f(x0) = {v:g}")
    # Ascending order lets each solve warm-start from the previous
    # point, which is the entire reason the barrier method is cheap:
    # consecutive centering problems differ only slightly.
    tv = np.sort(tv)
    rows, objs, slacks = [], [], []
    for ti in tv:
        x, _it, _ok = _centering(f0, fs, x, float(ti))
        rows.append(x.copy())
        objs.append(float(f0(x)))
        slacks.append([float(fi(x)) for fi in fs])
    path = np.array(rows)
    objs = np.array(objs)
    slack = np.array(slacks)
    m = len(fs)
    gap = m / tv
    return RichResult(
        title="Central path",
        summary_lines=[("n", int(path.shape[1])), ("constraints", int(m)),
                       ("points", int(tv.size)),
                       ("t range", f"{tv[0]:g} to {tv[-1]:g}"),
                       ("final gap", float(gap[-1]))],
        payload={
            "t": tv, "path": path, "objective": objs, "gap": gap,
            "slack": slack,
            "suboptimality_bounded": bool(
                np.all(objs - objs[-1] <= gap + 1e-09)),
            "strictly_feasible": bool(np.all(slack < 0)),
            "method": "boyd_central_path",
        },
    )


def cheatsheet():
    return "cvxcen: t->0 gives the ANALYTIC CENTRE (depends how constraints are written); t->inf gives the optimum"
