# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-form ridge via the augmented normal equation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_ridge_normal"]


def geron_ridge_normal(X, y, alpha, intercept_index=0):
    """
    Closed-form ridge via the augmented normal equation.

    Formula: theta = (X^T X + alpha A)^{-1} X^T y, A = diag(0, 1, ..., 1)

    Adding alpha to the diagonal makes the Gram matrix positive definite
    for any alpha > 0, so ridge has a solution even when X^T X is
    singular -- the property that makes it usable on collinear or wide
    designs where :func:`~morie.fn.hmneq.geron_normal_equation` refuses.
    The zero in A leaves the bias unpenalised.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    alpha : float
        Non-negative penalty weight.
    intercept_index : int or None, default 0
        Column that carries the bias and stays unpenalised.

    Returns
    -------
    result : RichResult
        Keys: theta, residuals, rss, effective_df, estimate, n, method.

    Examples
    --------
    alpha = 0 reproduces OLS on an exactly-fitting design:

    >>> X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    >>> [round(float(v), 9) + 0.0 for v in geron_ridge_normal(X, [1.0, 2.0, 3.0], 0.0)["theta"]]
    [0.0, 1.0]

    With alpha = 1 the hand-solved system [[3, 6], [6, 15]] theta = [6, 14]
    gives theta = (6/9, 6/9):

    >>> [round(float(v), 6) for v in geron_ridge_normal(X, [1.0, 2.0, 3.0], 1.0)["theta"]]
    [0.666667, 0.666667]

    NOTE the alpha conventions differ across the pair: the cost
    form penalises the MEAN squared error with (alpha/2)||theta||^2
    while hmridg's closed form adds alpha into the RSS normal
    equations, so the same nominal alpha shrinks differently; the
    cost-form equivalent of the closed form's alpha is 2*alpha/m.
    Found by the cross-language parity suite. Both are
    self-consistent; the mismatch is across the pair.

    References
    ----------
    Geron Ch 4
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_ridge_normal: X must be 2-D, got ndim={A.ndim}")
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if A.shape[0] == 0:
        raise ValueError("geron_ridge_normal: X has no rows")
    if yv.size != A.shape[0]:
        raise ValueError(f"geron_ridge_normal: X has {A.shape[0]} rows but y has {yv.size} entries")
    a = float(alpha)
    if not np.isfinite(a) or a < 0:
        raise ValueError(f"geron_ridge_normal: alpha must be finite and non-negative, got {alpha!r}")
    p = A.shape[1]
    d = np.ones(p)
    if intercept_index is not None:
        k = int(intercept_index)
        if not (0 <= k < p):
            raise ValueError(f"geron_ridge_normal: intercept_index {k} is outside {p} columns")
        d[k] = 0.0

    G = A.T @ A + a * np.diag(d)
    if np.linalg.matrix_rank(G) < p:
        raise ValueError(
            "geron_ridge_normal: X^T X + alpha A is singular; raise alpha or drop a duplicated column"
        )
    theta = np.linalg.solve(G, A.T @ yv)
    resid = A @ theta - yv
    # Effective degrees of freedom, trace of the hat matrix X (X^T X + alpha A)^-1 X^T.
    edf = float(np.trace(A @ np.linalg.solve(G, A.T)))
    return RichResult(
        title="Ridge by the augmented normal equation",
        summary_lines=[("alpha", a), ("RSS", float(resid @ resid)), ("Effective df", edf)],
        interpretation="Effective df falls from p towards 0 as alpha grows; that is the shrinkage made visible.",
        payload={
            "theta": theta,
            "residuals": resid,
            "rss": float(resid @ resid),
            "effective_df": edf,
            "alpha": a,
            "estimate": theta,
            "n": int(A.shape[0]),
            "method": "Ridge closed form (X^T X + alpha A) theta = X^T y",
        },
    )


def cheatsheet():
    return "hmridn: Closed-form ridge via augmented normal equation"
