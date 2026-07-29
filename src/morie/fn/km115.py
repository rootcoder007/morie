# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.3: BLEU-N, the geometric mean of n-gram precisions."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch8_bleu_n_geom_mean"]


def kamath_ch8_bleu_n_geom_mean(p_n, N=None):
    r"""BLEU-N = (prod_{n=1..N} p_n)^{1/N}.

    Computed in log space, so long products of small precisions do
    not underflow. A single zero precision makes the whole product
    zero -- that is BLEU's behaviour, and it is returned as 0.0.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.3, printed
    p. 323.

    Examples
    --------
    >>> out = kamath_ch8_bleu_n_geom_mean([0.5, 0.125])
    >>> round(out["estimate"], 12)      # sqrt(0.0625)
    0.25
    """
    p = np.atleast_1d(np.asarray(p_n, dtype=float))
    if p.size == 0:
        raise ValueError("no precisions given.")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("each p_n is a precision and must lie in "
                         "[0, 1].")
    if N is not None and int(N) != p.size:
        raise ValueError(
            f"N = {N} contradicts the {p.size} precisions given.")
    if np.any(p == 0):
        gm, logmean = 0.0, float("-inf")
    else:
        logmean = float(np.mean(np.log(p)))
        gm = float(np.exp(logmean))
    return RichResult(payload={
        "estimate": gm, "log_mean": logmean,
        "p_n": [float(v) for v in p], "n": int(p.size),
        "method": "BLEU-N geometric mean of precisions (Kamath Eq 8.3)"})


def cheatsheet():
    return "km115: (prod p_n)^(1/N), computed as exp(mean log p_n)"
