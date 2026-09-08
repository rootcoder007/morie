# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Linear score for class k in softmax regression."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsftm import geron_softmax_function

__all__ = ["geron_softmax_score"]


def geron_softmax_score(X, theta):
    """
    Linear score for class k in softmax regression.

    Formula: s_k(x) = theta_k^T x

    The scores are the raw linear part of softmax regression; the
    normalisation into probabilities is delegated to
    :func:`morie.fn.hmsftm.geron_softmax_function` rather than
    reimplemented here.

    Parameters
    ----------
    X : array-like
        Design matrix, shape (n, d); a 1-D input is read as one row.
    theta : array-like
        Parameter matrix, shape (d, K) with one column per class.

    Returns
    -------
    result : RichResult
        Keys: scores, p, predicted, estimate, n, method.

    Examples
    --------
    >>> r = geron_softmax_score([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]])
    >>> [float(v) for v in r["scores"][0]]
    [1.0, 2.0]
    >>> [round(float(v), 6) for v in r["p"][0]]
    [0.268941, 0.731059]
    >>> int(r["predicted"][0])
    1

    References
    ----------
    Géron Ch 4
    """
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim == 1:
        Xm = Xm.reshape(1, -1)
    if Xm.ndim != 2 or Xm.size == 0:
        raise ValueError("geron_softmax_score: X must be a non-empty 2-D design matrix")
    th = np.asarray(theta, dtype=float)
    if th.ndim == 1:
        th = th.reshape(-1, 1)
    if th.ndim != 2:
        raise ValueError("geron_softmax_score: theta must be a (d, K) matrix")
    if th.shape[0] != Xm.shape[1]:
        raise ValueError(
            f"geron_softmax_score: X has {Xm.shape[1]} features but theta has {th.shape[0]} rows; "
            "theta must be (n_features, n_classes)"
        )
    if th.shape[1] < 2:
        raise ValueError(f"geron_softmax_score: softmax regression needs >= 2 classes, got {th.shape[1]}")
    if not (np.all(np.isfinite(Xm)) and np.all(np.isfinite(th))):
        raise ValueError("geron_softmax_score: X and theta must be finite")

    scores = Xm @ th
    p = np.vstack([np.asarray(geron_softmax_function(row)["p"], dtype=float) for row in scores])
    pred = np.argmax(scores, axis=1)

    return RichResult(
        title="Softmax regression scores",
        summary_lines=[("Rows", int(Xm.shape[0])), ("Classes", int(th.shape[1]))],
        interpretation="Class scores are linear in x; softmax is monotone, so argmax score == argmax probability.",
        payload={
            "scores": scores,
            "p": p,
            "probabilities": p,
            "predicted": pred,
            "estimate": float(np.mean(np.max(p, axis=1))),
            "n": int(Xm.shape[0]),
            "method": "Linear class scores X @ theta, normalised via hmsftm",
        },
    )


def cheatsheet():
    return "hmsfts: Linear score for class k in softmax regression"
