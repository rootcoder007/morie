# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GloVe weighted least-squares cost over co-occurrence counts."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_glove_cost", "glove_weight"]


def glove_weight(x, x_max, alpha):
    """f(x) = (x / x_max)^alpha if x < x_max else 1."""
    x = np.asarray(x, dtype=float)
    return np.where(x < x_max, np.power(x / x_max, alpha), 1.0)


def kamath_glove_cost(X, W, W_tilde, b, b_tilde, x_max=100.0, alpha=0.75):
    """J = sum_{i,j} f(X_ij) (w_i . w~_j + b_i + b~_j - log X_ij)^2.

    Zero co-occurrences are SKIPPED, not weighted to zero after
    evaluating log 0: f(0) = 0 but 0 * -inf is nan, and GloVe is
    defined as a sum over the non-zero entries. How many entries were
    scored is reported.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 1,
    GloVe; the section is not present in the 2024 PDF, so the cost is
    implemented exactly as the spec line states (Pennington et al.
    2014).

    Examples
    --------
    >>> import math
    >>> J = kamath_glove_cost([[2.0]], [[0.0]], [[0.0]], [0.5], [0.0],
    ...                       x_max=100.0, alpha=0.75)
    >>> f = (2.0 / 100.0) ** 0.75
    >>> abs(J["estimate"] - f * (0.5 - math.log(2.0)) ** 2) < 1e-12
    True
    >>> J["n_nonzero"]
    1
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    W = np.atleast_2d(np.asarray(W, dtype=float))
    Wt = np.atleast_2d(np.asarray(W_tilde, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    bt = np.atleast_1d(np.asarray(b_tilde, dtype=float)).ravel()
    x_max = float(x_max)
    alpha = float(alpha)
    if x_max <= 0:
        raise ValueError("x_max must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if np.any(X < 0):
        raise ValueError("co-occurrence counts must be non-negative.")
    V, C = X.shape
    if W.shape[0] != V or Wt.shape[0] != C:
        raise ValueError(
            f"X is {V}x{C} but W has {W.shape[0]} rows and W_tilde "
            f"{Wt.shape[0]}.")
    if W.shape[1] != Wt.shape[1]:
        raise ValueError(
            f"word and context vectors differ in width: {W.shape[1]} "
            f"vs {Wt.shape[1]}.")
    if b.size != V or bt.size != C:
        raise ValueError(
            f"need {V} word biases and {C} context biases; got "
            f"{b.size} and {bt.size}.")

    nz = X > 0
    if not nz.any():
        raise ValueError(
            "every co-occurrence count is 0; the GloVe objective sums "
            "over the non-zero entries and there are none.")
    pred = W @ Wt.T + b[:, None] + bt[None, :]
    resid = np.zeros_like(pred)
    resid[nz] = pred[nz] - np.log(X[nz])
    f = np.zeros_like(pred)
    f[nz] = glove_weight(X[nz], x_max, alpha)
    terms = f * resid ** 2
    J = float(terms.sum())
    return RichResult(payload={
        "estimate": J, "cost": J,
        "weights": [[float(v) for v in row] for row in f],
        "residuals": [[float(v) for v in row] for row in resid],
        "n_nonzero": int(nz.sum()),
        "n": V * C,
        "method": "GloVe weighted least-squares cost"})


def cheatsheet():
    return "kmglv: sum f(X_ij)(w_i.w~_j + b_i + b~_j - log X_ij)^2"
