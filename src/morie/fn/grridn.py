# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-form ridge regression via the augmented normal equation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_ridge_normal_equation"]

_METHOD = "Ridge closed form (augmented normal equation)"


def geron_ridge_normal_equation(X, y, alpha, intercept=True):
    r"""Ridge estimate in closed form.

    .. math::
        \hat{\boldsymbol{\theta}}
          = (X^{\top}X + \alpha A)^{-1} X^{\top}\mathbf{y},
        \qquad A = \mathrm{diag}(0, 1, \dots, 1)

    The leading zero in :math:`A` is the whole subtlety: the bias term is
    not penalised, because shrinking it would make the fit depend on
    where the origin happens to sit.  Adding :math:`\alpha` to the
    remaining diagonal also makes the system solvable when
    :math:`X^{\top}X` is singular -- which is why ridge works where
    :mod:`morie.fn.grnorm` refuses.

    Parameters
    ----------
    X : array-like, shape (m, n)
        First column is treated as the bias when ``intercept=True``.
    y : array-like, shape (m,)
    alpha : float
        Non-negative penalty. ``alpha=0`` reproduces OLS.
    intercept : bool, optional
        Whether column 0 of ``X`` is the bias column.

    Returns
    -------
    RichResult
        Payload keys ``theta``, ``fitted``, ``residuals``, ``rss``,
        ``penalty``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-9 (Ridge closed form).

    Examples
    --------
    Collinear features that break OLS are fine here:

    >>> X = [[1.0, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 3.0, 3.0]]
    >>> r = geron_ridge_normal_equation(X, [2.0, 4.0, 6.0], alpha=1.0)
    >>> len(r["theta"])
    3

    Shrinkage is monotone in alpha (bias column excluded):

    >>> Xs = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    >>> a = geron_ridge_normal_equation(Xs, [4.0, 7.0, 10.0], alpha=0.0)["theta"][1]
    >>> b = geron_ridge_normal_equation(Xs, [4.0, 7.0, 10.0], alpha=10.0)["theta"][1]
    >>> round(a, 10), abs(b) < abs(a)
    (3.0, True)
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty (m, n) matrix, got shape {A.shape}.")
    if A.shape[0] != yv.size:
        raise ValueError(f"X has {A.shape[0]} rows but y has {yv.size} entries.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yv)):
        raise ValueError("X and y must be finite.")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be finite and non-negative, got {alpha}.")

    n = A.shape[1]
    Amat = np.eye(n)
    if intercept:
        if n < 2:
            raise ValueError("intercept=True needs at least one feature besides the bias column.")
        Amat[0, 0] = 0.0
    G = A.T @ A + alpha * Amat
    if np.linalg.matrix_rank(G) < n:
        raise ValueError(
            "X^T X + alpha*A is singular; increase alpha or drop the duplicated bias column."
        )
    theta = np.linalg.solve(G, A.T @ yv)
    fitted = A @ theta
    res = yv - fitted
    pen = float(alpha * (theta @ (Amat @ theta)))

    return RichResult(
        title="Ridge (closed form)",
        summary_lines=[("alpha", alpha), ("RSS", float(res @ res)), ("Penalty", pen)],
        payload={
            "theta": theta.tolist(),
            "fitted": fitted.tolist(),
            "residuals": res.tolist(),
            "rss": float(res @ res),
            "penalty": pen,
            "estimate": theta.tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grridn: theta = (X^T X + alpha diag(0,1..1))^-1 X^T y; bias column never penalised"
