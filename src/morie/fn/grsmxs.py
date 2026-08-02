# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Softmax class-score for class k in multinomial logistic regression."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_softmax_score", "score_matrix"]

_METHOD = "Softmax class scores s_k(x) = theta_k^T x"


def score_matrix(X, theta):
    """Return the ``(m, K)`` score matrix and the validated ``(X, Theta)``.

    ``theta`` is the ``(n, K)`` parameter matrix whose column ``k`` is
    :math:`\\theta_k`. Shared with the softmax probability/cost modules.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    T = np.asarray(theta, dtype=float)
    if T.ndim == 1:
        T = T[:, None]
    if X.ndim != 2 or X.size == 0:
        raise ValueError(f"X must be a non-empty (m, n) matrix, got shape {X.shape}.")
    if T.ndim != 2:
        raise ValueError(f"theta must be (n, K), got shape {T.shape}.")
    if T.shape[0] != X.shape[1]:
        raise ValueError(
            f"theta has {T.shape[0]} rows but X has {X.shape[1]} features; "
            "theta must be (n_features, K)."
        )
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(T)):
        raise ValueError("X and theta must be finite.")
    return X, T, X @ T


def geron_softmax_score(X, theta):
    r"""Linear score of every class for every instance.

    .. math::
        s_k(\mathbf{x}) = \boldsymbol{\theta}_k^{\top}\mathbf{x}

    Softmax regression is K linear models sharing one normalisation; this
    is the linear half, before any exponentiation.  Scores are defined only
    up to an additive constant per instance -- the softmax that follows
    cancels it.

    Parameters
    ----------
    X : array-like, shape (m, n)
    theta : array-like, shape (n, K)
        Column ``k`` is the parameter vector of class ``k``.

    Returns
    -------
    RichResult
        Payload keys ``scores`` (m x K), ``argmax`` (predicted class per
        row), ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-19 (Softmax score for class k).

    Examples
    --------
    Two instances, two features, three classes:

    >>> X = [[1.0, 2.0], [3.0, 4.0]]
    >>> T = [[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]
    >>> r = geron_softmax_score(X, T)
    >>> r["scores"]
    [[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]
    >>> r["argmax"]
    [1, 1]
    """
    X, T, S = score_matrix(X, theta)
    return RichResult(
        title="Softmax class scores",
        summary_lines=[("Instances", int(X.shape[0])), ("Classes", int(T.shape[1]))],
        payload={
            "scores": S.tolist(),
            "argmax": np.argmax(S, axis=1).astype(int).tolist(),
            "estimate": S.tolist(),
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsmxs: s_k(x) = theta_k^T x, the (m,K) score matrix X @ Theta before softmax"
