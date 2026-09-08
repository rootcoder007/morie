# morie.fn -- function file (rootcoder007/morie)
"""Novelty as self-information of an item."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["novelty"]


def novelty(item, popularity):
    """How surprising a recommendation is, in bits.

    Accuracy metrics reward recommending what everyone already likes, so
    a system optimised for them converges on the head of the catalogue.
    Novelty is the counterweight: the self-information of an item, which
    is large exactly when few users have seen it.  Reported in bits, so
    an item half the users know scores 1.

    Formula: ``nov(i) = -log2 P(i)``, with ``P(i)`` the share of the
    interactions that fell on item ``i``.

    Parameters
    ----------
    item : array-like
        Zero-based item indices whose novelty is wanted.
    popularity : array-like
        Interaction counts or probabilities per item; normalised here.

    Returns
    -------
    RichResult
        ``estimate`` (mean novelty over the supplied items), ``nov``,
        ``p``, ``n_items``.

    References
    ----------
    Vargas, S. & Castells, P. (2011).  Rank and relevance in novelty and
    diversity metrics for recommender systems.  RecSys 2011, 109-116.
    """
    pop = C.vec(popularity)
    tot = sum(pop)
    p = [t / tot for t in pop]
    idx = [int(round(v)) for v in C.vec(item)]
    nov = [(-math.log(p[i]) / math.log(2.0)) if p[i] > 0.0 else float("inf")
           for i in idx]
    return RichResult(payload={
        "estimate": sum(nov) / len(nov), "nov": nov, "p": p,
        "n_items": len(p), "method": "Novelty, self-information in bits"})


def cheatsheet():
    return "novlt: Novelty as self-information of an item."
