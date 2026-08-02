# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lasso regression cost, Géron Eq 4-11 (factor-of-two convention)."""

from . import _array_core as np

from ._richresult import RichResult
from .grlaso import geron_lasso_cost

__all__ = ["geron_ch4_lasso_regression_cost_function"]

_METHOD = "Lasso cost, 2*alpha convention (Eq 4-11)"


def geron_ch4_lasso_regression_cost_function(X, y, theta, alpha, penalize_intercept=False):
    r"""Géron Eq 4-11.

    .. math::
        J(\theta) = \mathrm{MSE}(\theta)
        + 2\alpha \sum_{i=1}^{n} |\theta_i|

    The only difference from Eq 4-10 (:mod:`morie.fn.grlaso`) is the
    factor of two on the penalty -- the edition writes the constant this
    way so that the subgradient comes out as :math:`2\alpha\,
    \mathrm{sign}(\theta)`, matching the ``2/m`` of the MSE gradient.
    So this function *is* ``grlaso`` evaluated at ``2 * alpha``, and it
    delegates there rather than re-deriving the sum.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    theta : array-like, shape (n,)
    alpha : float
        Non-negative; the effective L1 weight is ``2 * alpha``.
    penalize_intercept : bool, optional

    Returns
    -------
    RichResult
        Payload keys ``cost``, ``mse``, ``l1_penalty``, ``l1_norm``,
        ``alpha``, ``effective_alpha``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Geron (2026), Ch 4, Eq 4-11, p. 162.

    Examples
    --------
    Same data as ``grlaso``'s example, same ``alpha`` -- the penalty is
    exactly twice as large:

    >>> X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    >>> r = geron_ch4_lasso_regression_cost_function(X, [1.0, 2.0, 3.0],
    ...                                              [0.0, 1.0], alpha=0.5)
    >>> r["cost"]
    1.0
    >>> r["effective_alpha"]
    1.0
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative finite float, got {alpha}.")
    inner = geron_lasso_cost(X, y, theta, 2.0 * alpha, penalize_intercept=penalize_intercept)

    return RichResult(
        title="Lasso cost (Eq 4-11)",
        summary_lines=[("Cost", inner["cost"]), ("MSE", inner["mse"]),
                       ("2*alpha*L1", inner["l1_penalty"])],
        payload={
            "cost": inner["cost"],
            "mse": inner["mse"],
            "l1_penalty": inner["l1_penalty"],
            "l1_norm": inner["l1_norm"],
            "alpha": alpha,
            "effective_alpha": 2.0 * alpha,
            "estimate": inner["cost"],
            "n": inner["n"],
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grn011: J = MSE + 2*alpha*sum|theta_i| -- grlaso at 2*alpha (Eq 4-11)"
