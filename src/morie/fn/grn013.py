# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Elastic net cost, Géron Eq 4-13 (per-instance L2 convention)."""

import numpy as np

from ._richresult import RichResult
from .grlaso import geron_lasso_cost

__all__ = ["geron_ch4_elastic_net_cost_function"]

_METHOD = "Elastic net cost, Eq 4-13 convention"


def geron_ch4_elastic_net_cost_function(X, y, theta, alpha, r, penalize_intercept=False):
    r"""Géron Eq 4-13.

    .. math::
        J(\theta) = \mathrm{MSE}(\theta)
        + r\Bigl(2\alpha\sum_{i=1}^{n}|\theta_i|\Bigr)
        + (1-r)\Bigl(\frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2\Bigr)

    Two constants differ from Eq 4-12 (:mod:`morie.fn.grelas`): the L1
    arm carries a factor 2, and the L2 arm is divided by the number of
    instances ``m`` instead of 2.  The ``1/m`` matters -- with this
    convention the ridge arm keeps the same weight relative to the MSE
    as the dataset grows, so ``alpha`` transfers across sample sizes.

    The MSE and L1 arms delegate to
    :func:`morie.fn.grlaso.geron_lasso_cost` at weight ``2 * r * alpha``.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    theta : array-like, shape (n,)
    alpha : float
        Non-negative.
    r : float
        Mix ratio in ``[0, 1]``.
    penalize_intercept : bool, optional

    Returns
    -------
    RichResult
        Payload keys ``cost``, ``mse``, ``l1_penalty``, ``l2_penalty``,
        ``l1_norm``, ``l2_norm_sq``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Geron (2026), Ch 4, Eq 4-13, p. 165.

    Examples
    --------
    Three instances, perfect fit, slope 1, ``alpha = 1``, ``r = 0.5``:
    ``0 + 0.5*2*1 + 0.5*(1/3)*1``:

    >>> X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    >>> res = geron_ch4_elastic_net_cost_function(X, [1.0, 2.0, 3.0],
    ...                                           [0.0, 1.0], alpha=1.0, r=0.5)
    >>> round(res["cost"], 10)
    1.1666666667
    >>> round(res["l2_penalty"], 10)
    0.1666666667

    The ``1/m`` is real: the same parameters on six instances halve the
    ridge arm.

    >>> X6 = X + [[1.0, 4.0], [1.0, 5.0], [1.0, 6.0]]
    >>> y6 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> round(geron_ch4_elastic_net_cost_function(X6, y6, [0.0, 1.0],
    ...                                           alpha=1.0, r=0.5)["l2_penalty"], 10)
    0.0833333333
    """
    alpha = float(alpha)
    r = float(r)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative finite float, got {alpha}.")
    if not (0.0 <= r <= 1.0):
        raise ValueError(f"r is the L1 mix ratio and must lie in [0, 1], got {r}.")

    inner = geron_lasso_cost(X, y, theta, 2.0 * r * alpha, penalize_intercept=penalize_intercept)
    theta = np.asarray(theta, dtype=float).ravel()
    w = theta if penalize_intercept else theta[1:]
    m = inner["n"]
    l2sq = float(np.sum(w**2))
    l2_penalty = (1.0 - r) * (alpha / m) * l2sq
    cost = inner["cost"] + l2_penalty

    return RichResult(
        title="Elastic net cost (Eq 4-13)",
        summary_lines=[("Cost", cost), ("MSE", inner["mse"]),
                       ("L1 term", inner["l1_penalty"]), ("L2 term", l2_penalty)],
        payload={
            "cost": cost,
            "mse": inner["mse"],
            "l1_penalty": inner["l1_penalty"],
            "l2_penalty": l2_penalty,
            "l1_norm": inner["l1_norm"],
            "l2_norm_sq": l2sq,
            "alpha": alpha,
            "r": r,
            "estimate": cost,
            "n": m,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grn013: J = MSE + r*2*alpha*L1 + (1-r)*(alpha/m)*L2 -- Geron Eq 4-13"
