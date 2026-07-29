# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-form OLS via the normal equation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_normal_equation"]

_METHOD = "Normal equation (closed-form OLS)"


def geron_normal_equation(X, y, add_intercept=False, rcond=1e-12):
    r"""Solve least squares in one shot.

    .. math::
        \hat{\boldsymbol{\theta}} = (X^{\top}X)^{-1} X^{\top} \mathbf{y}

    Implemented as a *solve* of :math:`(X^{\top}X)\theta = X^{\top}y`
    rather than by forming the inverse: same answer, roughly half the
    work, and better conditioned.  When :math:`X^{\top}X` is singular --
    duplicated features, or more features than instances -- there is no
    unique solution and this raises instead of returning whatever the
    pseudo-inverse happens to pick.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix. Add your own bias column, or pass
        ``add_intercept=True``.
    y : array-like, shape (m,)
    add_intercept : bool, optional
        Prepend a column of ones.
    rcond : float, optional
        Reciprocal condition number below which ``X.T @ X`` is called
        singular.

    Returns
    -------
    RichResult
        Payload keys ``theta``, ``fitted``, ``residuals``, ``rss``,
        ``condition_number``, ``estimate`` (theta), ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-4 (Normal Equation).

    Examples
    --------
    Exactly linear data recovers the generating parameters:
    ``y = 4 + 3x``.

    >>> X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    >>> r = geron_normal_equation(X, [4.0, 7.0, 10.0])
    >>> [round(t, 10) for t in r["theta"]]
    [4.0, 3.0]
    >>> round(r["rss"], 12)
    0.0

    A duplicated column is not solvable in closed form:

    >>> try:
    ...     geron_normal_equation([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [1.0, 2.0, 3.0])
    ... except ValueError as exc:
    ...     print(str(exc).split(" (")[0])
    X^T X is singular
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty (m, n) matrix, got shape {A.shape}.")
    if add_intercept:
        A = np.hstack([np.ones((A.shape[0], 1)), A])
    if A.shape[0] != yv.size:
        raise ValueError(f"X has {A.shape[0]} rows but y has {yv.size} entries.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yv)):
        raise ValueError("X and y must be finite.")
    if A.shape[0] < A.shape[1]:
        raise ValueError(
            f"{A.shape[0]} instances for {A.shape[1]} parameters: X^T X cannot have full rank."
        )

    G = A.T @ A
    sv = np.linalg.svd(G, compute_uv=False)
    rc = float(sv.min() / sv.max()) if sv.max() > 0 else 0.0
    if rc < rcond:
        raise ValueError(
            f"X^T X is singular (reciprocal condition number {rc:.1e} < {rcond:g}); "
            "features are collinear."
        )
    theta = np.linalg.solve(G, A.T @ yv)
    fitted = A @ theta
    res = yv - fitted

    return RichResult(
        title="Normal equation",
        summary_lines=[("Parameters", int(theta.size)), ("RSS", float(res @ res))],
        payload={
            "theta": theta.tolist(),
            "fitted": fitted.tolist(),
            "residuals": res.tolist(),
            "rss": float(res @ res),
            "condition_number": float(sv.max() / sv.min()),
            "estimate": theta.tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grnorm: theta = (X^T X)^-1 X^T y, solved not inverted; collinear X raises"
