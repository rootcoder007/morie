# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Softmax probabilities from a linear score model."""

import numpy as np

from ._richresult import RichResult
from .grn021 import softmax_vector
from .grsmxs import score_matrix

__all__ = ["geron_softmax_probability", "probability_matrix"]

_METHOD = "Softmax regression class probabilities"


def probability_matrix(X, theta):
    """Row-wise softmax of ``X @ theta``; shared with the cost/gradient modules."""
    X, T, S = score_matrix(X, theta)
    P = np.vstack([softmax_vector(row) for row in S])
    return X, T, P


def geron_softmax_probability(X, theta):
    r"""Class probabilities of softmax (multinomial logistic) regression.

    .. math::
        \hat p_k = \frac{\exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K}\exp(s_j(\mathbf{x}))}

    Scores come from :mod:`morie.fn.grsmxs` and the normalised exponential
    from :mod:`morie.fn.grn021` -- this module is the composition, not a
    third copy of either.

    Parameters
    ----------
    X : array-like, shape (m, n)
    theta : array-like, shape (n, K)

    Returns
    -------
    RichResult
        Payload keys ``probabilities`` (m x K), ``predictions``,
        ``scores``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-20 (Softmax function).

    Examples
    --------
    Scores ``[1, 2, 1]`` for the first instance give
    ``e/(2e + e^2) = 0.2119`` to classes 0 and 2:

    >>> X = [[1.0, 2.0], [3.0, 4.0]]
    >>> T = [[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]
    >>> r = geron_softmax_probability(X, T)
    >>> [round(v, 6) for v in r["probabilities"][0]]
    [0.211942, 0.576117, 0.211942]
    >>> r["predictions"]
    [1, 1]
    >>> round(sum(r["probabilities"][1]), 12)
    1.0
    """
    X, T, P = probability_matrix(X, theta)
    _, _, S = score_matrix(X, theta)
    return RichResult(
        title="Softmax probabilities",
        summary_lines=[("Instances", int(X.shape[0])), ("Classes", int(P.shape[1]))],
        payload={
            "probabilities": P.tolist(),
            "predictions": np.argmax(P, axis=1).astype(int).tolist(),
            "scores": S.tolist(),
            "estimate": P.tolist(),
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsmxp: p_hat = softmax(X @ Theta) row-wise; composes grsmxs scores with grn021 softmax"
