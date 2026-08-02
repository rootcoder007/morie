# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Elastic net cost: MSE plus a mix of L1 and L2 penalties."""

from . import _array_core as np

from ._richresult import RichResult
from .grlaso import geron_lasso_cost

__all__ = ["geron_elastic_net_cost"]

_METHOD = "Elastic net cost (Eq 4-12)"


def geron_elastic_net_cost(X, y, theta, alpha, r, penalize_intercept=False):
    r"""Géron Eq 4-12.

    .. math::
        J(\theta) = \mathrm{MSE}(\theta)
        + r\,\alpha \sum_{i=1}^{n}|\theta_i|
        + \frac{1-r}{2}\,\alpha \sum_{i=1}^{n}\theta_i^2

    ``r`` slides between the two extremes: ``r = 1`` is exactly lasso,
    ``r = 0`` is exactly ridge.  Both endpoints are reachable and are
    worth checking -- the L1 arm of an elastic net is the only one that
    can drive a coefficient to precisely zero.

    The MSE and L1 halves are delegated to
    :func:`morie.fn.grlaso.geron_lasso_cost` (called at weight
    ``r * alpha``); only the L2 arm is computed here.  As in ``grlaso``
    the intercept ``theta[0]`` is excluded from both penalties.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    theta : array-like, shape (n,)
    alpha : float
        Non-negative overall regularisation strength.
    r : float
        Mix ratio in ``[0, 1]``; ``r = 1`` is lasso, ``r = 0`` is ridge.
    penalize_intercept : bool, optional

    Returns
    -------
    RichResult
        Payload keys ``cost``, ``mse``, ``l1_penalty``, ``l2_penalty``,
        ``l1_norm``, ``l2_norm_sq``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-12 (Elastic Net cost function).

    Examples
    --------
    Perfect fit, slope 1, ``alpha = 1``, half-and-half mix: the cost is
    ``0 + 0.5*1*1 + (0.5/2)*1*1``:

    >>> X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    >>> r_ = geron_elastic_net_cost(X, [1.0, 2.0, 3.0], [0.0, 1.0], alpha=1.0, r=0.5)
    >>> r_["cost"]
    0.75
    >>> (r_["l1_penalty"], r_["l2_penalty"])
    (0.5, 0.25)

    At ``r = 1`` the L2 arm is switched off and elastic net collapses to
    lasso:

    >>> geron_elastic_net_cost(X, [1.0, 2.0, 3.0], [0.0, 1.0], alpha=1.0, r=1.0)["cost"]
    1.0
    """
    alpha = float(alpha)
    r = float(r)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative finite float, got {alpha}.")
    if not (0.0 <= r <= 1.0):
        raise ValueError(f"r is the L1 mix ratio and must lie in [0, 1], got {r}.")

    inner = geron_lasso_cost(X, y, theta, r * alpha, penalize_intercept=penalize_intercept)
    theta = np.asarray(theta, dtype=float).ravel()
    w = theta if penalize_intercept else theta[1:]
    l2sq = float(np.sum(w**2))
    l2_penalty = ((1.0 - r) / 2.0) * alpha * l2sq
    cost = inner["cost"] + l2_penalty

    return RichResult(
        title="Elastic net cost",
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
            "n": inner["n"],
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grelas: J = MSE + r*alpha*L1 + ((1-r)/2)*alpha*L2 -- Geron Eq 4-12"
