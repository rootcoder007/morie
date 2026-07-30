# morie.fn -- function file (rootcoder007/morie)
"""Regression estimator with auxiliary."""

import numpy as np

from ._richresult import RichResult

__all__ = ["regression_estimator_multi", "regression_estimator"]


def regression_estimator_multi(y, x, X_mean, weights=None):
    r"""Multivariate regression estimator:

    .. math:: \hat{\bar Y}_{reg} = \bar y
              + \mathbf b'(\bar{\mathbf X} - \bar{\mathbf x}),

    the several-auxiliary form of :mod:`morie.fn.regest`.

    Adding auxiliaries cannot reduce the fitted :math:`R^2`, but it
    can and does increase the VARIANCE of the estimator, because
    every extra coefficient is itself estimated from the same
    sample. The gain is :math:`(1 - R^2)` while the cost grows with
    :math:`p/n`; with many weak auxiliaries the second term wins.
    Both are returned so the trade is visible rather than assumed
    favourable.

    Parameters
    ----------
    y : array-like, shape (n,)
        Study variable.
    x : array-like, shape (n, p)
        Auxiliaries.
    X_mean : array-like, shape (p,)
        Known population means of the auxiliaries.
    weights : array-like, optional
        Design weights.

    Returns
    -------
    RichResult
        keys: ``mean``, ``coefficients``, ``R2``,
        ``variance_ratio_to_simple_mean``, ``p_over_n``,
        ``adjustment``, ``n``, ``p``, ``method``.
    """
    from ._survey import check_weights

    yv = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    n, p = X.shape
    if n <= p + 1:
        raise ValueError(f"need more observations than auxiliaries, got "
                         f"{n} and {p}.")
    Xm = np.atleast_1d(np.asarray(X_mean, dtype=float)).ravel()
    if Xm.size != p:
        raise ValueError(f"X_mean has {Xm.size} entries for {p} auxiliaries.")
    w = np.ones(n) if weights is None else check_weights(weights, n)
    sw = float(w.sum())
    xbar = (w[:, None] * X).sum(axis=0) / sw
    ybar = float(np.sum(w * yv) / sw)
    Xc = X - xbar
    yc = yv - ybar
    A = (w[:, None] * Xc).T @ Xc
    b = np.linalg.pinv(A) @ ((w[:, None] * Xc).T @ yc)
    fit = Xc @ b
    ss_tot = float(np.sum(w * yc ** 2))
    r2 = float(np.sum(w * fit ** 2) / ss_tot) if ss_tot > 0 else 0.0
    return RichResult(payload={
        "mean": ybar + float(b @ (Xm - xbar)),
        "coefficients": b, "R2": r2,
        "variance_ratio_to_simple_mean": float(1.0 - r2),
        "p_over_n": float(p) / float(n),
        "adjustment": float(b @ (Xm - xbar)),
        "tradeoff": "gain is (1 - R^2); cost grows with p/n, so weak extra "
                    "auxiliaries can make the estimator worse",
        "n": int(n), "p": int(p),
        "method": "Multivariate regression estimator; more auxiliaries is not automatically better"})


def cheatsheet():
    return "reglmd: extra auxiliaries always raise R^2 and can still raise the VARIANCE"


#: Catalogue alias for :func:`regression_estimator_multi`.
regression_estimator = regression_estimator_multi
