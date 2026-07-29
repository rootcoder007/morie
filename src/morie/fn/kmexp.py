# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 6: canary exposure as a memorization measure."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_memorization_exposure"]


def kamath_memorization_exposure(canary_ll, candidate_lls):
    r"""exposure = log2(|Candidates|) - log2(rank(canary)).

    ``candidate_lls`` are the model log-likelihoods of the other
    candidate strings in the canary's randomness space; the canary
    itself completes that space, so |Candidates| is one more than the
    number given. Rank 1 (the canary is the single most likely string)
    gives the maximum exposure log2(|Candidates|); a canary sitting at
    chance gets about 1 bit. Candidates that TIE with the canary are
    counted as ahead of it, which is the conservative reading.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Memorization
    Exposure; Carlini et al. (2019).

    Examples
    --------
    >>> out = kamath_memorization_exposure(-1.0, [-2.0, -3.0, -0.5])
    >>> out["rank"], out["estimate"]      # log2(4) - log2(2)
    (2, 1.0)
    """
    c = float(canary_ll)
    others = np.atleast_1d(np.asarray(candidate_lls, dtype=float))
    if others.size == 0:
        raise ValueError("no competing candidates were given; a rank "
                         "in a space of one is not informative.")
    if not (np.isfinite(c) and np.all(np.isfinite(others))):
        raise ValueError("log-likelihoods must be finite.")
    total = int(others.size) + 1
    rank = 1 + int(np.sum(others >= c))
    exposure = math.log2(total) - math.log2(rank)
    return RichResult(payload={
        "estimate": exposure, "exposure": exposure, "rank": rank,
        "n_candidates": total,
        "max_exposure": math.log2(total),
        "n": total,
        "method": "canary exposure (Kamath Ch 6)"})


def cheatsheet():
    return "kmexp: log2(candidate space) - log2(the canary's rank)"
