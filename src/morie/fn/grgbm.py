# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient boosting residual-fitting step (squared loss)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_gradient_boosting_residual"]

_METHOD = "Gradient boosting residual step (squared loss)"


def _best_stump(X, r):
    """Least-squares decision stump: one feature, one threshold."""
    m, n = X.shape
    best = None
    for j in range(n):
        vals = np.unique(X[:, j])
        if vals.size < 2:
            continue
        cuts = (vals[:-1] + vals[1:]) / 2.0
        for t in cuts:
            left = X[:, j] <= t
            if not left.any() or left.all():
                continue
            lv, rv = r[left].mean(), r[~left].mean()
            sse = float(((r[left] - lv) ** 2).sum() + ((r[~left] - rv) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, j, float(t), float(lv), float(rv))
    if best is None:
        raise ValueError(
            "no split is possible: every feature takes a single value, so a stump "
            "cannot separate any instances."
        )
    return best


def geron_gradient_boosting_residual(X, y, F_prev, learner=None, learning_rate=1.0):
    r"""One boosting stage: fit the residual, then add it in.

    .. math::
        r_i^{(m)} = y_i - F_{m-1}(x_i),\qquad
        F_m = F_{m-1} + \nu\, h_m,\quad h_m \text{ fitted on } (x_i, r_i)

    Under squared loss the residual *is* the negative gradient
    (:math:`-\partial L/\partial F = y - F` up to the factor 2), which
    is what makes "fit the residual" and "take a gradient step in
    function space" the same algorithm.

    ``learner`` may be any callable ``learner(X, r) -> predictions``;
    its output is checked for shape and finiteness before it is used.
    With ``learner=None`` a least-squares decision stump is fitted here
    -- the weak learner Géron uses to introduce boosting.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    F_prev : array-like, shape (m,) or scalar
        Current ensemble prediction.
    learner : callable, optional
        ``learner(X, residuals) -> array-like of shape (m,)``.
    learning_rate : float, optional
        Shrinkage :math:`\nu` in ``(0, 1]``, default 1.

    Returns
    -------
    RichResult
        Payload keys ``residuals``, ``h_prediction``, ``F_new``,
        ``mse_before``, ``mse_after``, ``stump`` (``None`` when a custom
        learner was supplied), ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 6, Gradient Boosting section.

    Examples
    --------
    A step function is learned in one stage by the default stump:

    >>> X = [[1.0], [2.0], [3.0], [4.0]]
    >>> r = geron_gradient_boosting_residual(X, [0.0, 0.0, 1.0, 1.0], 0.0)
    >>> r["residuals"]
    [0.0, 0.0, 1.0, 1.0]
    >>> r["F_new"]
    [0.0, 0.0, 1.0, 1.0]
    >>> r["stump"]["threshold"]
    2.5
    >>> (r["mse_before"], r["mse_after"])
    (0.5, 0.0)

    Shrinkage moves only part of the way, which is the point of it:

    >>> r2 = geron_gradient_boosting_residual(X, [0.0, 0.0, 1.0, 1.0], 0.0,
    ...                                       learning_rate=0.1)
    >>> r2["F_new"]
    [0.0, 0.0, 0.1, 0.1]
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if A.ndim != 2:
        raise ValueError(f"X must be 2-D of shape (m, n), got shape {A.shape}.")
    m = A.shape[0]
    if y.size != m:
        raise ValueError(f"y has {y.size} entries but X has {m} rows.")
    F = np.asarray(F_prev, dtype=float).ravel()
    if F.size == 1:
        F = np.full(m, float(F[0]))
    if F.size != m:
        raise ValueError(f"F_prev has {F.size} entries but X has {m} rows.")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(y)) and np.all(np.isfinite(F))):
        raise ValueError("X, y and F_prev must all be finite.")
    nu = float(learning_rate)
    if not (0.0 < nu <= 1.0):
        raise ValueError(f"learning_rate must lie in (0, 1], got {nu}.")

    resid = y - F
    stump = None
    if learner is None:
        sse, j, t, lv, rv = _best_stump(A, resid)
        h = np.where(A[:, j] <= t, lv, rv)
        stump = {"feature": int(j), "threshold": t, "left_value": lv,
                 "right_value": rv, "sse": sse}
    else:
        if not callable(learner):
            raise ValueError(f"learner must be callable, got {type(learner).__name__}.")
        h = np.asarray(learner(A, resid), dtype=float).ravel()
        if h.size != m:
            raise ValueError(
                f"learner returned {h.size} predictions but there are {m} instances."
            )
        if not np.all(np.isfinite(h)):
            raise ValueError("learner returned non-finite predictions.")

    F_new = F + nu * h

    return RichResult(
        title="Gradient boosting stage",
        summary_lines=[("MSE before", float(np.mean(resid**2))),
                       ("MSE after", float(np.mean((y - F_new) ** 2))),
                       ("Shrinkage", nu)],
        payload={
            "residuals": resid.tolist(),
            "h_prediction": h.tolist(),
            "F_new": F_new.tolist(),
            "mse_before": float(np.mean(resid**2)),
            "mse_after": float(np.mean((y - F_new) ** 2)),
            "stump": stump,
            "learning_rate": nu,
            "estimate": F_new.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grgbm: r = y - F_{m-1}; fit h on r (stump by default); F_m = F_{m-1} + nu*h"
