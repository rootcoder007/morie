# morie.fn -- function file (rootcoder007/morie)
"""Divergent transition count and rate."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['divrate', 'divergent_transitions_count']


def divrate(divergent):
    """Divergent transition count and rate.

    A divergence is not a tuning nuisance to be squashed by shrinking the step size: it marks the region of the target the sampler could not reach, so the draws are biased in a direction the divergences themselves identify. Any non-zero count is reported as such -- the flag is ``any``, not a rate threshold -- because the paper's point is that even a few divergences are sensitive identifiers of pathology.


    Formula: rate = n_divergent / n_total

    Parameters
    ----------
    divergent : array-like
        Per-iteration 0/1 divergence indicators; a list of lists is one chain per row.

    Returns
    -------
    RichResult
        ``count``, ``rate``, ``per_chain``, ``per_chain_rate``, ``any``, ``n``.

    References
    ----------
    Betancourt (2017), A Conceptual Introduction to Hamiltonian Monte
    Carlo, arXiv:1701.02434, Section 6.2: divergent transitions are
    'extremely sensitive identifiers' of the pathological neighbourhoods
    a trajectory failed to explore.  Verified against the paper.
    """
    D = divergent
    if not (isinstance(D, (list, tuple)) and D and isinstance(D[0], (list, tuple))):
        D = [C.vec(D)]
    else:
        D = [C.vec(row) for row in D]
    per = [sum(1 for v in row if v != 0) for row in D]
    tot = sum(len(row) for row in D)
    cnt = sum(per)
    return RichResult(payload={
        "count": cnt, "rate": cnt / tot if tot else float("nan"),
        "per_chain": per,
        "per_chain_rate": [per[i] / len(D[i]) for i in range(len(D))],
        "any": cnt > 0, "n": tot,
        "method": "Divergent transition count and rate"})


divergent_transitions_count = divrate


def cheatsheet():
    return "divcd: Divergent transition count and rate."
