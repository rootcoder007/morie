# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.22: the equalising log-probability-ratio regulariser."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_log_prob_ratio_attr"]


def kamath_ch6_log_prob_ratio_attr(a_i, a_j, K=None, lam=1.0):
    """R = lam (1/K) sum_{k=1..K} log[P(a_i^(k)) / P(a_j^(k))].

    Zero exactly when the two groups' attribute words are predicted
    with equal probability -- that equality is the objective. ``a_i``
    and ``a_j`` hold the K paired softmax probabilities; ``K`` is
    checked against their length when given.

    NOTE the printed Eq 6.22 carries the scaling lam and the 1/K
    average; both are implemented, with lam defaulting to 1, so the
    bare sum-of-log-ratios is ``unweighted``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.22, printed
    p. 244.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch6_log_prob_ratio_attr([0.5, 0.5], [0.25, 0.5], 2)
    >>> abs(out["estimate"] - math.log(2.0) / 2) < 1e-12
    True
    >>> kamath_ch6_log_prob_ratio_attr([0.4, 0.6], [0.4, 0.6])["estimate"]
    0.0
    """
    pi = np.atleast_1d(np.asarray(a_i, dtype=float))
    pj = np.atleast_1d(np.asarray(a_j, dtype=float))
    if pi.size == 0:
        raise ValueError("a_i is empty; a mean over no word pairs is "
                         "undefined, not 0.")
    if pi.shape != pj.shape:
        raise ValueError(
            f"a_i has {pi.size} probabilities but a_j has {pj.size}; the "
            "words must be paired.")
    if np.any(pi <= 0) or np.any(pj <= 0) or np.any(pi > 1) or np.any(pj > 1):
        raise ValueError("every probability must lie in (0, 1]; a zero "
                         "makes the log ratio undefined.")
    if K is not None and int(K) != pi.size:
        raise ValueError(
            f"K = {int(K)} contradicts the {pi.size} pairs supplied.")
    lam = float(lam)
    if not np.isfinite(lam):
        raise ValueError("lam must be finite.")
    logs = np.log(pi / pj)
    return RichResult(payload={
        "estimate": float(lam * logs.mean()),
        "unweighted": float(logs.sum()),
        "per_pair": [float(v) for v in logs], "lam": lam,
        "K": int(pi.size), "n": int(pi.size),
        "method": "equalising log-probability-ratio regulariser "
                  "(Kamath Eq 6.22)"})


def cheatsheet():
    return "km098: lam * mean log P(a_i)/P(a_j) over K word pairs"
