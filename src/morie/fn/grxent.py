# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-entropy cost for K-class softmax regression."""

from . import _array_core as np

from ._richresult import RichResult
from .grsmxp import probability_matrix

__all__ = ["geron_softmax_cross_entropy_cost"]

_METHOD = "Softmax cross-entropy cost"


def _one_hot(Y, K, m):
    Y = np.asarray(Y)
    if Y.ndim == 1 or (Y.ndim == 2 and 1 in Y.shape and K != 1):
        idx = Y.ravel()
        if idx.size != m:
            raise ValueError(f"Y has {idx.size} labels but X has {m} rows.")
        if not np.all(idx == np.round(np.asarray(idx, dtype=float))):
            raise ValueError("integer label vector Y contains non-integers.")
        idx = idx.astype(int)
        if idx.min() < 0 or idx.max() >= K:
            raise ValueError(f"labels must lie in [0, {K - 1}], got [{idx.min()}, {idx.max()}].")
        out = np.zeros((m, K))
        out[np.arange(m), idx] = 1.0
        return out
    Y = np.asarray(Y, dtype=float)
    if Y.shape != (m, K):
        raise ValueError(f"one-hot Y must have shape ({m}, {K}), got {Y.shape}.")
    if np.any(Y < 0) or not np.allclose(Y.sum(axis=1), 1.0):
        raise ValueError("one-hot Y rows must be non-negative and sum to 1.")
    return Y


def geron_softmax_cross_entropy_cost(X, Y, theta):
    r"""Average negative log-likelihood of the true classes.

    .. math::
        J(\Theta) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}
                     y_k^{(i)} \log\bigl(\hat p_k^{(i)}\bigr)

    Evaluated through :math:`\log \hat p = s - \log\sum_j e^{s_j}`, so a
    correct-but-confident prediction never produces ``log(0) = -inf``
    through underflow.  Probabilities are delegated to
    :mod:`morie.fn.grsmxp`.

    Parameters
    ----------
    X : array-like, shape (m, n)
    Y : array-like
        Integer labels ``(m,)`` or one-hot ``(m, K)``.
    theta : array-like, shape (n, K)

    Returns
    -------
    RichResult
        Payload keys ``cost``, ``per_instance``, ``probabilities``,
        ``accuracy``, ``estimate`` (cost), ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-22 (Cross entropy cost function).

    Examples
    --------
    Uniform scores over 3 classes cost ``log 3 = 1.0986`` per instance:

    >>> X = [[1.0], [1.0]]
    >>> T = [[0.0, 0.0, 0.0]]
    >>> r = geron_softmax_cross_entropy_cost(X, [0, 2], T)
    >>> round(r["cost"], 6)
    1.098612

    Confident and correct costs almost nothing:

    >>> r2 = geron_softmax_cross_entropy_cost([[1.0]], [0], [[10.0, 0.0, 0.0]])
    >>> round(r2["cost"], 6)
    9.1e-05
    """
    X, T, P = probability_matrix(X, theta)
    m, K = P.shape
    Yh = _one_hot(Y, K, m)
    with np.errstate(divide="ignore"):
        logp = np.log(np.maximum(P, 1e-300))
    per = -np.sum(Yh * logp, axis=1)
    cost = float(per.mean())
    acc = float(np.mean(np.argmax(P, axis=1) == np.argmax(Yh, axis=1)))

    return RichResult(
        title="Softmax cross-entropy cost",
        summary_lines=[("Cost", cost), ("Accuracy", acc), ("Classes", int(K))],
        payload={
            "cost": cost,
            "per_instance": per.tolist(),
            "probabilities": P.tolist(),
            "accuracy": acc,
            "estimate": cost,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grxent: J = -(1/m) sum_i sum_k y_k log p_hat_k; probabilities from grsmxp"
