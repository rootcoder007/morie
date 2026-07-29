# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lasso (L1) regression cost function."""

import numpy as np

from ._richresult import RichResult
from .grmse import geron_linreg_mse_cost

__all__ = ["geron_lasso_cost"]

_METHOD = "Lasso (L1) cost"


def geron_lasso_cost(X, y, theta, alpha, penalize_intercept=False):
    r"""Géron Eq 4-10.

    .. math::
        J(\theta) = \mathrm{MSE}(\theta)
        + \alpha \sum_{i=1}^{n} |\theta_i|

    The sum starts at :math:`i = 1`: the bias :math:`\theta_0` is not
    penalised, because shrinking it would make the fit depend on where
    the origin of ``y`` happens to sit.  Pass
    ``penalize_intercept=True`` only if ``X`` genuinely has no bias
    column.

    The data-fit term is delegated to
    :func:`morie.fn.grmse.geron_linreg_mse_cost`.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    theta : array-like, shape (n,)
        ``theta[0]`` is treated as the bias unless
        ``penalize_intercept`` is set.
    alpha : float
        Non-negative regularisation strength.
    penalize_intercept : bool, optional
        Include ``theta[0]`` in the L1 sum. Default ``False``.

    Returns
    -------
    RichResult
        Payload keys ``cost``, ``mse``, ``l1_penalty``, ``l1_norm``,
        ``n_zero``, ``estimate`` (= ``cost``), ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-10 (Lasso cost function).

    Examples
    --------
    Perfect fit, one unit-size slope, ``alpha = 0.5``: the cost is
    entirely penalty, ``0 + 0.5 * 1``:

    >>> X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    >>> r = geron_lasso_cost(X, [1.0, 2.0, 3.0], [0.0, 1.0], alpha=0.5)
    >>> r["cost"]
    0.5
    >>> r["mse"]
    0.0

    The intercept escapes the penalty -- moving it (and the targets with
    it) leaves the L1 term untouched:

    >>> r2 = geron_lasso_cost(X, [3.0, 4.0, 5.0], [2.0, 1.0], alpha=0.5)
    >>> r2["l1_penalty"]
    0.5
    """
    fit = geron_linreg_mse_cost(X, y, theta)          # validates shapes
    theta = np.asarray(theta, dtype=float).ravel()
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative finite float, got {alpha}.")

    w = theta if penalize_intercept else theta[1:]
    if w.size == 0 and not penalize_intercept:
        raise ValueError(
            "theta has only the intercept, which is not penalised; the lasso "
            "term would be empty. Pass penalize_intercept=True if X has no bias column."
        )
    l1 = float(np.sum(np.abs(w)))
    penalty = alpha * l1
    cost = fit["cost"] + penalty

    return RichResult(
        title="Lasso cost",
        summary_lines=[("Cost", cost), ("MSE", fit["cost"]), ("alpha * L1", penalty)],
        payload={
            "cost": cost,
            "mse": fit["cost"],
            "l1_penalty": penalty,
            "l1_norm": l1,
            "n_zero": int(np.sum(w == 0.0)),
            "alpha": alpha,
            "estimate": cost,
            "n": fit["n"],
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grlaso: J = MSE + alpha*sum_{i>=1}|theta_i| -- Geron Eq 4-10"
