# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mean squared error cost for linear regression (theta minimizes MSE)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_linreg_mse_cost"]

_METHOD = "Linear-regression MSE cost"


def geron_linreg_mse_cost(X, y, theta):
    r"""MSE cost of a linear hypothesis, Géron Eq 4-3.

    .. math::
        \mathrm{MSE}(\theta) = \frac{1}{m}\sum_{i=1}^{m}
        \bigl(\theta^{\mathsf T} x^{(i)} - y^{(i)}\bigr)^2

    ``X`` is used exactly as supplied -- no bias column is prepended.
    If the model has an intercept, ``X`` must already carry the column
    of ones and ``theta[0]`` is its weight.

    This is the cost that :func:`morie.fn.grn005.geron_ch4_normal_equation`
    minimises in closed form and that
    :func:`morie.fn.grn007.geron_ch4_mse_gradient_vector` differentiates;
    the regularised costs in ``grlaso``, ``grelas``, ``grn011`` and
    ``grn013`` all delegate their data-fit term here.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix, one row per instance.
    y : array-like, shape (m,)
        Targets.
    theta : array-like, shape (n,)
        Parameter vector.

    Returns
    -------
    RichResult
        Payload keys ``cost``, ``rmse``, ``residuals``, ``predictions``,
        ``estimate`` (= ``cost``), ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-3 (MSE cost function).

    Examples
    --------
    A hypothesis biased high by exactly 0.5 on every instance has
    ``MSE = 0.25`` -- the squared bias, no variance:

    >>> X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    >>> r = geron_linreg_mse_cost(X, [1.0, 2.0, 3.0], [0.5, 1.0])
    >>> r["cost"]
    0.25
    >>> r["rmse"]
    0.5

    The exact fit costs nothing:

    >>> geron_linreg_mse_cost(X, [1.0, 2.0, 3.0], [0.0, 1.0])["cost"]
    0.0
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D of shape (m, n), got shape {X.shape}.")
    m, n = X.shape
    if m == 0:
        raise ValueError("X has no rows; MSE over zero instances is undefined.")
    if y.size != m:
        raise ValueError(f"y has {y.size} entries but X has {m} rows.")
    if theta.size != n:
        raise ValueError(
            f"theta has {theta.size} entries but X has {n} columns; prepend the "
            f"bias column to X yourself if the model has an intercept."
        )
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(theta)):
        raise ValueError("X, y and theta must all be finite.")

    pred = X @ theta
    resid = pred - y
    cost = float(np.mean(resid**2))

    return RichResult(
        title="Linear regression MSE cost",
        summary_lines=[("MSE", cost), ("RMSE", float(np.sqrt(cost))), ("m", int(m))],
        payload={
            "cost": cost,
            "rmse": float(np.sqrt(cost)),
            "residuals": resid.tolist(),
            "predictions": pred.tolist(),
            "estimate": cost,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmse: MSE(theta) = mean((X theta - y)^2) -- Geron Eq 4-3"
