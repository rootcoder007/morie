# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Soft voting ensemble prediction (argmax of the mean probability)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_soft_voting"]

_METHOD = "Soft voting ensemble"


def geron_soft_voting(probabilities, weights=None):
    r"""Average the predicted probabilities, then take the argmax.

    .. math::
        \hat y = \arg\max_k \frac{1}{L}\sum_{l=1}^{L} p_l(y = k \mid x)

    Soft voting usually beats hard voting for the reason visible in the
    numbers: a classifier that is 95% sure counts more than one that is
    51% sure, whereas hard voting throws that confidence away.  The cost
    is that it only works when the probabilities are comparable across
    models -- an overconfident member dominates the average whether or
    not it is right.  Rows are checked to be genuine distributions for
    that reason.

    Parameters
    ----------
    probabilities : array-like, shape (L, m, K) or (L, K)
        Per-classifier class probabilities.
    weights : array-like, shape (L,), optional
        Non-negative classifier weights; default uniform.

    Returns
    -------
    RichResult
        Payload keys ``y_hat``, ``mean_probabilities``, ``confidence``,
        ``margin`` (top minus runner-up), ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 6, Voting Classifier (soft) section.

    Examples
    --------
    One confident vote for class 1 outweighs two lukewarm votes for
    class 0 -- the case where soft and hard voting disagree:

    >>> P = [[[0.55, 0.45]], [[0.55, 0.45]], [[0.05, 0.95]]]
    >>> r = geron_soft_voting(P)
    >>> [round(v, 6) for v in r["mean_probabilities"][0]]
    [0.383333, 0.616667]
    >>> r["y_hat"]
    [1]

    Weights are honoured:

    >>> w = geron_soft_voting(P, weights=[5.0, 5.0, 1.0])
    >>> w["y_hat"]
    [0]
    """
    A = np.asarray(probabilities, dtype=float)
    if A.ndim == 2:
        A = A[:, None, :]
    if A.ndim != 3 or A.size == 0:
        raise ValueError(
            f"probabilities must be (L, m, K) or (L, K), got shape {A.shape}."
        )
    if not np.all(np.isfinite(A)):
        raise ValueError("probabilities contains non-finite values.")
    if np.any(A < 0):
        raise ValueError("probabilities must be non-negative.")
    if not np.allclose(A.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError("each classifier's rows must sum to 1; these are not probabilities.")
    L = A.shape[0]

    if weights is None:
        w = np.ones(L) / L
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != L:
            raise ValueError(f"weights has {w.size} entries but there are {L} classifiers.")
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative with a positive sum.")
        w = w / w.sum()

    M = np.tensordot(w, A, axes=(0, 0))
    yhat = np.argmax(M, axis=1)
    srt = np.sort(M, axis=1)
    margin = srt[:, -1] - (srt[:, -2] if M.shape[1] > 1 else 0.0)

    return RichResult(
        title="Soft voting",
        summary_lines=[("Voters", int(L)), ("Instances", int(M.shape[0]))],
        payload={
            "y_hat": yhat.astype(int).tolist(),
            "mean_probabilities": M.tolist(),
            "confidence": M.max(axis=1).tolist(),
            "margin": np.atleast_1d(margin).tolist(),
            "estimate": yhat.astype(int).tolist(),
            "n": int(M.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grvots: y_hat = argmax mean_l p_l(k|x); confidence-weighted, so it beats hard voting when calibrated"
