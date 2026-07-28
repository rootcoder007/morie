# morie.fn -- function file (rootcoder007/morie)
"""Counting lemma: objects into distinguishable cells."""

from math import comb

from ._richresult import RichResult

__all__ = ["gibbons_distributing_objects"]


def gibbons_distributing_objects(n, r):
    r"""Lemma 3.2.1: the number of distinguishable ways to distribute
    n like objects into r distinguishable cells with no cell empty is

    .. math:: \binom{n - 1}{r - 1},

    the stars-and-bars count with each cell forced non-empty. This is
    the combinatorial engine behind every runs distribution in the
    chapter: a run IS a non-empty cell of consecutive like elements.

    Parameters
    ----------
    n : int
        Number of like objects, n >= 1.
    r : int
        Number of cells, 1 <= r <= n.

    Returns
    -------
    RichResult
        keys: ``count``, ``n``, ``r``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Lemma 3.2.1.
    """
    n, r = int(n), int(r)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    if not 1 <= r <= n:
        raise ValueError(f"r must lie in 1..{n}, got {r}.")
    return RichResult(
        payload={
            "count": comb(n - 1, r - 1), "n": n, "r": r,
            "method": "C(n-1, r-1) non-empty distributions (Gibbons Lemma 3.2.1)",
        }
    )


def cheatsheet():
    return "gb321l: C(n-1, r-1); a run is a non-empty cell"
