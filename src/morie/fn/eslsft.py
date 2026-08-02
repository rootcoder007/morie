# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Softmax transformation (ESL Ch 4.4 / 11.3)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_softmax"]


def esl_softmax(T):
    """
    Softmax p_k = exp(T_k) / sum_l exp(T_l).

    Computed as exp(T - max T) / sum exp(T - max T), which is
    algebraically identical because the shift cancels top and bottom,
    but does not overflow: exp(1000) is inf in float64 while the
    shifted form is exact. The transformation is invariant to adding
    a constant to every score, so the scores are NOT identifiable
    without a constraint -- ESL's convention pins the last class at
    zero, and the payload reports the shift applied so that
    non-identifiability is visible rather than hidden.

    Parameters
    ----------
    T : array-like
        Scores for K classes, either a vector or (n, K) rows.

    Returns
    -------
    result : dict
        Keys: estimate (probability of the first class of the first
        row), probabilities (row-major), argmax (0-based per row),
        max_shift, n, K, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 4.4 (Eq. 4.17) and
    Ch 11.3.

    Examples
    --------
    >>> out = esl_softmax([0.0, 0.0])
    >>> out["probabilities"]
    [0.5, 0.5]
    >>> import math
    >>> out = esl_softmax([1.0, 0.0])
    >>> abs(out["estimate"] - math.e / (math.e + 1)) < 1e-15
    True
    >>> esl_softmax([1000.0, 999.0])["estimate"] == esl_softmax([1.0, 0.0])["estimate"]
    True
    >>> esl_softmax([[0.0, 0.0], [5.0, 0.0]])["argmax"]
    [0, 0]
    """
    T = np.asarray(T, dtype=float)
    single_row = T.ndim == 1
    M = np.atleast_2d(T)
    n, K = M.shape
    if K < 1:
        raise ValueError("softmax needs at least one class.")
    if not np.all(np.isfinite(M)):
        raise ValueError("scores must be finite.")
    shift = M.max(axis=1, keepdims=True)
    e = np.exp(M - shift)
    P = e / e.sum(axis=1, keepdims=True)
    return RichResult(payload={
        "estimate": float(P[0, 0]),
        "probabilities": [float(v) for v in (P[0] if single_row else P.ravel())],
        "argmax": [int(v) for v in np.argmax(P, axis=1)],
        "max_shift": [float(v) for v in shift.ravel()],
        "n": int(n), "K": int(K),
        "method": "softmax via max-shift; invariant to adding a constant to all scores"})


def cheatsheet():
    return "eslsft: exp(T-max)/sum exp(T-max); shift-invariant, so scores unidentifiable"
