# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.8: the Log-Probability Bias Score."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_lpbs_bias"]


def _pair_probs(p, name):
    v = np.atleast_1d(np.asarray(
        [float(x) for x in (p.values() if isinstance(p, dict) else p)],
        dtype=float))
    if v.size != 2:
        raise ValueError(
            f"{name} must hold exactly two probabilities, one per social "
            f"group; got {v.size}.")
    if np.any(v <= 0) or np.any(v > 1):
        raise ValueError(
            f"every entry of {name} must lie in (0, 1]; a zero makes the "
            "log ratio undefined.")
    return v


def kamath_ch6_lpbs_bias(p_a, p_prior):
    """LPBS = log(p_ai / p_prior_i) - log(p_aj / p_prior_j).

    Each group's target probability is normalised by ITS OWN prior --
    that is what stops the score from simply re-measuring how common
    the group word is. 0 means the two groups are boosted equally by
    the attribute; the sign says which group the attribute favours.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.8, printed
    p. 235.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch6_lpbs_bias([0.5, 0.25], [0.25, 0.25])
    >>> abs(out["estimate"] - math.log(2.0)) < 1e-12
    True
    >>> kamath_ch6_lpbs_bias([0.5, 0.5], [0.25, 0.25])["estimate"]
    0.0
    """
    pa = _pair_probs(p_a, "p_a")
    pp = _pair_probs(p_prior, "p_prior")
    logs = np.log(pa / pp)
    return RichResult(payload={
        "estimate": float(logs[0] - logs[1]),
        "normalised_log_i": float(logs[0]),
        "normalised_log_j": float(logs[1]), "n": 2,
        "method": "Log-Probability Bias Score (Kamath Eq 6.8)"})


def cheatsheet():
    return "km084: difference of prior-normalised log probabilities"
