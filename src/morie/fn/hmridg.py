# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ridge (L2) regression cost."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_ridge_cost"]


def geron_ridge_cost(X, y, theta, alpha, intercept_index=0):
    """
    Ridge (L2) regression cost.

    Formula: J = MSE + alpha * (1/2) sum theta_i^2

    The bias term is NOT penalised -- shrinking it would make the fit
    depend on where the target's zero happens to sit. ``intercept_index``
    says which entry of theta is the bias (set it to None to penalise
    everything). The gradient (2/m) X^T (X theta - y) + alpha * A theta
    is returned with it, A being the identity with a zero in the bias
    slot.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    theta : array-like, shape (n,)
    alpha : float
        Non-negative penalty weight; 0 recovers plain OLS.
    intercept_index : int or None, default 0
        Unpenalised coefficient.

    Returns
    -------
    result : RichResult
        Keys: cost, mse, penalty, gradient, estimate, n, method.

    Examples
    --------
    A perfect fit leaves only the penalty, (1/2) * alpha * 1^2:

    >>> r = geron_ridge_cost([[1.0, 1.0], [1.0, 2.0]], [1.0, 2.0], [0.0, 1.0], alpha=1.0)
    >>> float(r["mse"]), float(r["penalty"]), float(r["cost"])
    (0.0, 0.5, 0.5)

    With alpha = 0 the cost is the plain MSE:

    >>> float(geron_ridge_cost([[1.0, 1.0]], [3.0], [1.0, 1.0], alpha=0.0)["cost"])
    1.0

    The gradient of the penalty alone is alpha * theta off the intercept:

    >>> [float(v) for v in r["gradient"]]
    [0.0, 1.0]

    NOTE the alpha conventions differ across the pair: the cost
    form penalises the MEAN squared error with (alpha/2)||theta||^2
    while hmridn's closed form adds alpha into the RSS normal
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
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    th = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()
    m = A.shape[0]
    if m == 0:
        raise ValueError("geron_ridge_cost: X has no rows")
    if yv.size != m:
        raise ValueError(f"geron_ridge_cost: X has {m} rows but y has {yv.size} entries")
    if th.size != A.shape[1]:
        raise ValueError(f"geron_ridge_cost: theta has {th.size} entries but X has {A.shape[1]} columns")
    a = float(alpha)
    if not np.isfinite(a) or a < 0:
        raise ValueError(f"geron_ridge_cost: alpha must be finite and non-negative, got {alpha!r}")

    mask = np.ones_like(th)
    if intercept_index is not None:
        k = int(intercept_index)
        if not (0 <= k < th.size):
            raise ValueError(f"geron_ridge_cost: intercept_index {k} is outside theta of length {th.size}")
        mask[k] = 0.0

    resid = A @ th - yv
    mse = float(np.mean(resid**2))
    penalty = float(0.5 * a * np.sum(mask * th**2))
    grad = (2.0 / m) * (A.T @ resid) + a * mask * th
    return RichResult(
        title="Ridge cost",
        summary_lines=[("MSE", mse), ("L2 penalty", penalty), ("Total", mse + penalty)],
        interpretation="alpha trades fit for shrinkage; the bias term is left unpenalised on purpose.",
        payload={
            "cost": mse + penalty,
            "mse": mse,
            "penalty": penalty,
            "gradient": grad,
            "alpha": a,
            "estimate": mse + penalty,
            "n": int(m),
            "method": "Ridge cost MSE + (alpha/2)||theta||^2 with an unpenalised bias",
        },
    )


def cheatsheet():
    return "hmridg: Ridge (L2) regression cost"
