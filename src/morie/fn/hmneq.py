# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-form OLS via the normal equation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_normal_equation"]


def geron_normal_equation(X, y, fit_intercept=False):
    """
    Closed-form OLS via the normal equation.

    Formula: theta_hat = (X^T X)^{-1} X^T y

    Setting the gradient of the MSE to zero gives X^T X theta = X^T y,
    which is solved here by a Cholesky-style linear solve rather than by
    forming the inverse (same answer, better conditioned). If X^T X is
    singular the parameters are NOT identified and a ValueError is
    raised: silently returning a pseudo-inverse answer would hide a rank
    deficiency the user needs to know about. The condition number of
    X^T X is reported because it squares the condition number of X, the
    reason Geron prefers SVD for wide or collinear designs.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Feature matrix; include your own column of ones, or set
        ``fit_intercept``.
    y : array-like, shape (m,)
        Targets.
    fit_intercept : bool, default False
        Prepend a column of ones.

    Returns
    -------
    result : RichResult
        Keys: theta, residuals, rss, cond, estimate, n, method.

    Examples
    --------
    >>> r = geron_normal_equation([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], [1.0, 2.0, 3.0])
    >>> [round(float(v), 9) + 0.0 for v in r["theta"]]
    [0.0, 1.0]
    >>> round(float(r["rss"]), 12)
    0.0

    A slope-only fit through (1, 2) and (2, 4):

    >>> float(geron_normal_equation([[1.0], [2.0]], [2.0, 4.0])["theta"][0])
    2.0

    References
    ----------
    Geron Ch 4
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_normal_equation: X must be 2-D, got ndim={A.ndim}")
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if A.shape[0] == 0:
        raise ValueError("geron_normal_equation: X has no rows")
    if yv.size != A.shape[0]:
        raise ValueError(f"geron_normal_equation: X has {A.shape[0]} rows but y has {yv.size} entries")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yv)):
        raise ValueError("geron_normal_equation: inputs contain non-finite values")
    if fit_intercept:
        A = np.hstack([np.ones((A.shape[0], 1)), A])
    if A.shape[0] < A.shape[1]:
        raise ValueError(
            f"geron_normal_equation: {A.shape[0]} rows for {A.shape[1]} parameters -- X^T X is singular by shape"
        )

    G = A.T @ A
    rank = int(np.linalg.matrix_rank(G))
    if rank < A.shape[1]:
        raise ValueError(
            f"geron_normal_equation: X^T X has rank {rank} for {A.shape[1]} columns; "
            "the columns are collinear and theta is not identified"
        )
    theta = np.linalg.solve(G, A.T @ yv)
    resid = A @ theta - yv
    return RichResult(
        title="OLS by the normal equation",
        summary_lines=[("Parameters", int(theta.size)), ("RSS", float(resid @ resid))],
        interpretation="cond(X^T X) is the square of cond(X); above ~1e8 prefer an SVD/lstsq route.",
        payload={
            "theta": theta,
            "residuals": resid,
            "rss": float(resid @ resid),
            "cond": float(np.linalg.cond(G)),
            "estimate": theta,
            "n": int(A.shape[0]),
            "method": "Normal equation solved as a linear system (no explicit inverse)",
        },
    )


def cheatsheet():
    return "hmneq: Closed-form OLS via the normal equation"
