# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-entropy gradient vector for a single class k."""

from . import _array_core as np

from ._richresult import RichResult
from .grxeng import gradient_matrix

__all__ = ["geron_ch4_cross_entropy_gradient_vector"]

_METHOD = "Cross-entropy gradient vector for one class"


def geron_ch4_cross_entropy_gradient_vector(X, Y, Theta, k):
    r"""One column of the softmax cross-entropy gradient.

    .. math::
        \nabla_{\boldsymbol{\theta}_k} J(\Theta)
          = \frac{1}{m}\sum_{i=1}^{m}
            \bigl(\hat p_k^{(i)} - y_k^{(i)}\bigr)\mathbf{x}^{(i)}

    The full ``(n, K)`` gradient is computed once by
    :func:`morie.fn.grxeng.gradient_matrix`; this module selects class
    ``k`` from it rather than recomputing the softmax K times, which is
    the whole reason Géron writes the batched form in Eq 4-23.

    Parameters
    ----------
    X : array-like, shape (m, n)
    Y : array-like
        Integer labels ``(m,)`` or one-hot ``(m, K)``.
    Theta : array-like, shape (n, K)
    k : int
        Class whose gradient vector is wanted.

    Returns
    -------
    RichResult
        Payload keys ``gradient`` (length n), ``class``,
        ``gradient_norm``, ``mean_error`` (mean of
        :math:`\hat p_k - y_k`), ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron (2026), Ch 4, Eq 4-24, p. 176.

    Examples
    --------
    True class is 0 and the model is uniform over 3 classes, so class 0
    is under-predicted by 2/3:

    >>> r = geron_ch4_cross_entropy_gradient_vector([[1.0]], [0], [[0.0, 0.0, 0.0]], k=0)
    >>> [round(v, 6) for v in r["gradient"]]
    [-0.666667]
    >>> round(r["mean_error"], 6)
    -0.666667

    A class nobody belongs to has a positive gradient -- push its scores
    down:

    >>> r2 = geron_ch4_cross_entropy_gradient_vector([[1.0]], [0], [[0.0, 0.0, 0.0]], k=1)
    >>> round(r2["gradient"][0], 6)
    0.333333
    """
    X, P, Yh, G = gradient_matrix(X, Y, Theta)
    k = int(k)
    if not (0 <= k < G.shape[1]):
        raise ValueError(f"k must lie in [0, {G.shape[1] - 1}], got {k}.")
    g = G[:, k]

    return RichResult(
        title="Cross-entropy gradient vector",
        summary_lines=[("Class", k), ("Gradient norm", float(np.linalg.norm(g)))],
        payload={
            "gradient": g.tolist(),
            "class": k,
            "gradient_norm": float(np.linalg.norm(g)),
            "mean_error": float(np.mean(P[:, k] - Yh[:, k])),
            "estimate": g.tolist(),
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grn024: column k of (1/m) X^T (P_hat - Y); selects from grxeng rather than recomputing"
