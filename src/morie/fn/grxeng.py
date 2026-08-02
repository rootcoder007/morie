# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient of K-class softmax cross-entropy w.r.t. Theta."""

from . import _array_core as np

from ._richresult import RichResult
from .grsmxp import probability_matrix
from .grxent import _one_hot

__all__ = ["geron_softmax_cost_gradient", "gradient_matrix"]

_METHOD = "Softmax cross-entropy gradient"


def gradient_matrix(X, Y, theta):
    """Return ``(X, P, Yh, G)`` with ``G`` the ``(n, K)`` gradient."""
    X, T, P = probability_matrix(X, theta)
    m, K = P.shape
    Yh = _one_hot(Y, K, m)
    G = (X.T @ (P - Yh)) / m
    return X, P, Yh, G


def geron_softmax_cost_gradient(X, Y, theta):
    r"""Gradient of the cross-entropy cost for every class at once.

    .. math::
        \nabla_{\boldsymbol{\theta}_k} J(\Theta)
          = \frac{1}{m}\sum_{i=1}^{m}
            \bigl(\hat p_k^{(i)} - y_k^{(i)}\bigr)\mathbf{x}^{(i)}

    The softmax and the log cancel: what survives is "predicted minus
    observed, weighted by the inputs" -- the same shape as the linear
    regression gradient.  The columns of the gradient sum to zero
    because the probabilities do, so the cost is flat along a uniform
    shift of all class parameters.  Probabilities come from
    :mod:`morie.fn.grsmxp`, labels are validated by
    :mod:`morie.fn.grxent`.

    Parameters
    ----------
    X : array-like, shape (m, n)
    Y : array-like
        Integer labels ``(m,)`` or one-hot ``(m, K)``.
    theta : array-like, shape (n, K)

    Returns
    -------
    RichResult
        Payload keys ``gradient`` (n x K), ``probabilities``,
        ``gradient_norm``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-23 (Cross-entropy gradient).

    Examples
    --------
    One instance ``x = [1]``, uniform scores, true class 0: predicted
    minus observed is ``[1/3 - 1, 1/3, 1/3]``.

    >>> r = geron_softmax_cost_gradient([[1.0]], [0], [[0.0, 0.0, 0.0]])
    >>> [round(v, 6) for v in r["gradient"][0]]
    [-0.666667, 0.333333, 0.333333]

    Gradient columns sum to zero for every feature:

    >>> abs(sum(r["gradient"][0])) < 1e-12
    True
    """
    X, P, Yh, G = gradient_matrix(X, Y, theta)
    return RichResult(
        title="Softmax cost gradient",
        summary_lines=[("Instances", int(X.shape[0])), ("Gradient norm", float(np.linalg.norm(G)))],
        payload={
            "gradient": G.tolist(),
            "probabilities": P.tolist(),
            "gradient_norm": float(np.linalg.norm(G)),
            "estimate": G.tolist(),
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grxeng: grad_Theta J = (1/m) X^T (P_hat - Y); columns sum to zero"
