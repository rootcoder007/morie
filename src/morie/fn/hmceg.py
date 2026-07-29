# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient of cross-entropy for softmax regression."""

import numpy as np

from ._richresult import RichResult
from .hmcec import geron_cross_entropy_cost

__all__ = ["geron_cross_entropy_gradient"]


def geron_cross_entropy_gradient(X, Y, theta):
    """
    Gradient of cross-entropy for softmax regression.

    Formula: grad_{theta_k} J = (1/m) sum_i (p_ik - y_ik) x_i

    The forward pass (probabilities, cost, label handling) is delegated to
    :func:`morie.fn.hmcec.geron_cross_entropy_cost` so the two modules can
    never disagree about what the softmax is; only the gradient
    ``(1/m) X^T (P - Y)`` is computed here.

    Because the softmax is over-parameterised, the gradient columns sum to
    zero for every feature -- ``column_sum_max_abs`` is reported as a
    check on that identity.

    Parameters
    ----------
    X : array-like, shape (m, n)
    Y : array-like, shape (m, K) or (m,)
        One-hot targets or integer labels.
    theta : array-like, shape (n, K)

    Returns
    -------
    result : RichResult
        Keys: gradient, probabilities, cost, grad_norm,
        column_sum_max_abs, estimate, n, method.

    Examples
    --------
    Zero parameters, one sample of class 0: p = (0.5, 0.5) so the
    gradient is ``x (p - y) = x (-0.5, 0.5)``.

    >>> r = geron_cross_entropy_gradient([[1.0, 2.0]], [0], [[0.0, 0.0], [0.0, 0.0]])
    >>> [[round(v, 9) for v in row] for row in r["gradient"]]
    [[-0.5, 0.5], [-1.0, 1.0]]
    >>> round(r["column_sum_max_abs"], 12)
    0.0

    A perfectly confident correct prediction has a vanishing gradient:

    >>> r2 = geron_cross_entropy_gradient([[1.0]], [0], [[50.0, 0.0]])
    >>> [round(v, 9) for v in r2["gradient"][0]]
    [0.0, 0.0]

    References
    ----------
    Géron Ch 4
    """
    fwd = geron_cross_entropy_cost(X, Y, theta)
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    P = np.asarray(fwd["probabilities"], dtype=float)
    m, K = P.shape

    Yarr = np.asarray(Y)
    if Yarr.ndim == 1 or (Yarr.ndim == 2 and Yarr.size == m and K != 1 and 1 in Yarr.shape):
        Yoh = np.zeros((m, K))
        Yoh[np.arange(m), Yarr.ravel().astype(int)] = 1.0
    else:
        Yoh = np.asarray(Yarr, dtype=float).reshape(m, K)

    G = (Xa.T @ (P - Yoh)) / m
    colsum = float(np.max(np.abs(G.sum(axis=1))))

    return RichResult(
        title="Softmax cross-entropy gradient",
        summary_lines=[("Cost", float(fwd["cost"])), ("Gradient norm", float(np.linalg.norm(G)))],
        interpretation="Softmax is over-parameterised: each row of the gradient sums to zero.",
        payload={
            "gradient": G.tolist(),
            "grad": G,
            "probabilities": P.tolist(),
            "cost": float(fwd["cost"]),
            "grad_norm": float(np.linalg.norm(G)),
            "column_sum_max_abs": colsum,
            "estimate": float(np.linalg.norm(G)),
            "n": int(m),
            "method": "softmax cross-entropy gradient (1/m) X^T (P - Y); forward pass delegated to hmcec",
        },
    )


def cheatsheet():
    return "hmceg: Gradient of cross-entropy for softmax regression"
