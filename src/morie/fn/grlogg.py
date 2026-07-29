# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient of the logistic-regression cost."""

import numpy as np

from ._richresult import RichResult
from .grlogp import geron_logistic_regression_probability

__all__ = ["geron_logistic_cost_gradient"]

_METHOD = "Logistic cost gradient (Eq 4-18)"


def geron_logistic_cost_gradient(X, y, theta):
    r"""Géron Eq 4-18.

    .. math::
        \frac{\partial J(\theta)}{\partial \theta_j}
        = \frac{1}{m}\sum_{i=1}^{m}
        \bigl(\sigma(\theta^{\mathsf T}x^{(i)}) - y^{(i)}\bigr)\,x_j^{(i)}

    The same shape as the linear-regression gradient
    (:mod:`morie.fn.grn007`) up to the constant: prediction error times
    feature, averaged.  The sigmoid and the log in the cost cancel each
    other in the derivative, which is why the logistic gradient has no
    :math:`\sigma'` factor in it.

    Probabilities are delegated to
    :func:`morie.fn.grlogp.geron_logistic_regression_probability`.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
        Labels, 0 or 1.
    theta : array-like, shape (n,)

    Returns
    -------
    RichResult
        Payload keys ``gradient``, ``grad_norm``, ``probabilities``,
        ``errors``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-18 (Logistic cost partial derivative).

    Examples
    --------
    Two identical rows, opposite labels, ``theta = 0``: the errors
    ``+0.5`` and ``-0.5`` cancel and the gradient is zero:

    >>> X = [[1.0], [1.0]]
    >>> g = geron_logistic_cost_gradient(X, [1.0, 0.0], [0.0])["gradient"]
    >>> [round(v, 12) for v in g]
    [0.0]

    Two negatives instead: the model is 0.5 on both, so the gradient is
    ``+0.5`` and gradient descent pushes the logit down:

    >>> geron_logistic_cost_gradient(X, [0.0, 0.0], [0.0])["gradient"]
    [0.5]
    """
    probs = geron_logistic_regression_probability(X, theta)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    p = np.asarray(probs["probability"], dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if y.size != p.size:
        raise ValueError(f"y has {y.size} entries but X has {p.size} rows.")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y must contain only 0 and 1 for the logistic gradient.")

    err = p - y
    m = p.size
    grad = (X.T @ err) / m

    return RichResult(
        title="Logistic cost gradient",
        summary_lines=[("||grad||", float(np.linalg.norm(grad))), ("m", int(m))],
        payload={
            "gradient": grad.tolist(),
            "grad_norm": float(np.linalg.norm(grad)),
            "probabilities": p.tolist(),
            "errors": err.tolist(),
            "estimate": grad.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grlogg: dJ/dtheta = (1/m) X^T (sigmoid(X theta) - y) -- Geron Eq 4-18"
