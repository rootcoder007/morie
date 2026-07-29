# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Softmax (normalized exponential) turning class scores into probabilities."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_ch4_softmax_function", "softmax_vector"]

_METHOD = "Softmax (normalized exponential)"


def softmax_vector(s):
    """Stable softmax of a 1-D score vector (shared by the softmax family).

    Subtracting the maximum before exponentiating leaves the ratio
    unchanged and bounds every exponent at 0.
    """
    s = np.asarray(s, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("score vector is empty.")
    if not np.all(np.isfinite(s)):
        raise ValueError("score vector contains non-finite values.")
    e = np.exp(s - s.max())
    return e / e.sum()


def geron_ch4_softmax_function(s, k, K=None):
    r"""Probability the softmax assigns to class ``k``.

    .. math::
        \hat p_k = \sigma(\mathbf{s}(x))_k
                 = \frac{\exp(s_k(x))}{\sum_{j=1}^{K} \exp(s_j(x))}

    Parameters
    ----------
    s : array-like, shape (K,)
        Class scores (logits).
    k : int
        Index of the class whose probability is wanted, ``0 <= k < K``.
    K : int, optional
        Declared number of classes; checked against ``len(s)`` when given.

    Returns
    -------
    RichResult
        Payload keys ``probability`` (class ``k``), ``probabilities``
        (all ``K``), ``argmax``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron (2026), Ch 4, Eq 4-21, p. 174.

    Examples
    --------
    Scores ``[0, 1, 2]``: denominators ``1 + e + e^2 = 11.107``, so class 2
    takes ``e^2/11.107``:

    >>> r = geron_ch4_softmax_function([0.0, 1.0, 2.0], k=2, K=3)
    >>> round(r["probability"], 6)
    0.665241
    >>> round(sum(r["probabilities"]), 12)
    1.0

    Adding a constant to every score changes nothing:

    >>> r2 = geron_ch4_softmax_function([100.0, 101.0, 102.0], k=2, K=3)
    >>> round(r2["probability"], 6)
    0.665241
    """
    p = softmax_vector(s)
    if K is not None:
        K = int(K)
        if K != p.size:
            raise ValueError(f"K={K} but the score vector has {p.size} entries.")
    k = int(k)
    if not (0 <= k < p.size):
        raise ValueError(f"k must lie in [0, {p.size - 1}], got {k}.")

    return RichResult(
        title="Softmax function",
        summary_lines=[("Class", k), ("p_hat_k", float(p[k]))],
        payload={
            "probability": float(p[k]),
            "probabilities": p.tolist(),
            "argmax": int(np.argmax(p)),
            "estimate": float(p[k]),
            "n": int(p.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grn021: softmax p_k = exp(s_k)/sum_j exp(s_j); max-shifted for stability, shift-invariant"
