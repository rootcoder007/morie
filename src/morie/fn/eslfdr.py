# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benjamini-Hochberg FDR control (ESL Ch 18.7)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_bh_fdr"]


def esl_bh_fdr(pvalues, alpha):
    """
    Benjamini-Hochberg step-up procedure.

    Formula: order p_(1) <= ... <= p_(m); find the LARGEST j with
    p_(j) <= j alpha / m and reject H_(1) .. H_(j). Rejecting the
    largest such j (not merely every p below its own line) is what
    makes the procedure a step-UP rule, and getting that wrong quietly
    under-rejects. Rejection indices refer to the ORIGINAL input
    order and are 0-based.

    Parameters
    ----------
    pvalues : array-like
        P-values in [0, 1], at least one.
    alpha : float
        Target false discovery rate in (0, 1).

    Returns
    -------
    result : dict
        Keys: estimate (number rejected), rejected (0-based indices
        in input order), threshold (largest rejected p, nan if none),
        cutoff_rank (1-based j, 0 if none), m, alpha, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 18.7.1 (Alg. 18.2);
    Benjamini & Hochberg (1995).

    Examples
    --------
    Sorted p = 0.001, 0.02, 0.3, 0.9 against the lines j*0.1/4 =
    0.025, 0.05, 0.075, 0.1: ranks 1 and 2 clear, so two hypotheses
    are rejected -- the ones at input positions 0 and 2.

    >>> out = esl_bh_fdr([0.001, 0.3, 0.02, 0.9], 0.1)
    >>> out["estimate"]
    2
    >>> out["rejected"]
    [0, 2]
    >>> out["cutoff_rank"]
    2
    >>> out2 = esl_bh_fdr([0.01, 0.02, 0.03, 0.04], 0.1)
    >>> out2["estimate"]
    4
    >>> esl_bh_fdr([0.5, 0.6], 0.05)["cutoff_rank"]
    0
    >>> esl_bh_fdr([0.5, 1.5], 0.05)
    Traceback (most recent call last):
        ...
    ValueError: p-values must lie in [0, 1]; got 1.5.
    """
    p = np.atleast_1d(np.asarray(pvalues, dtype=float))
    alpha = float(alpha)
    m = p.size
    if m == 0:
        raise ValueError("the BH procedure needs at least one p-value.")
    bad = p[(p < 0) | (p > 1)]
    if bad.size:
        raise ValueError(f"p-values must lie in [0, 1]; got {float(bad[0])}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")
    order = np.argsort(p, kind="stable")
    ps = p[order]
    ranks = np.arange(1, m + 1)
    under = np.flatnonzero(ps <= ranks * alpha / m)
    if under.size == 0:
        return RichResult(payload={
            "estimate": 0, "rejected": [], "threshold": float("nan"),
            "cutoff_rank": 0, "m": int(m), "alpha": alpha,
            "method": "Benjamini-Hochberg step-up; no rejections"})
    j = int(under[-1])                      # largest j meeting the line
    rejected = sorted(int(v) for v in order[: j + 1])
    return RichResult(payload={
        "estimate": len(rejected), "rejected": rejected,
        "threshold": float(ps[j]), "cutoff_rank": j + 1,
        "m": int(m), "alpha": alpha,
        "method": "Benjamini-Hochberg step-up: reject through the largest j with p_(j) <= j alpha/m"})


def cheatsheet():
    return "eslfdr: reject ranks 1..j for the LARGEST j with p_(j) <= j alpha/m"


# compact alias per ledger/NAMING.md
eslbhfdr = esl_bh_fdr
