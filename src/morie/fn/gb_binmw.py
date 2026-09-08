# morie.fn -- function file (rootcoder007/morie)
"""W-U linkage for the two-sample rank statistics."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_mw_binomial_link"]


def gibbons_mw_binomial_link(W, m):
    r"""The constant link between the rank-sum and Mann-Whitney forms
    (Gibbons Ch. 6.6):

    .. math:: U = W - \frac{m(m + 1)}{2}.

    The two statistics are the SAME test: W counts ranks, U counts
    (X, Y) pairs with X > Y, and the offset m(m+1)/2 is the rank sum
    a sample of m earns just by existing. Any p-value computed from
    one applies verbatim to the other.

    Parameters
    ----------
    W : float
        Rank sum of the size-m sample; must be at least m(m+1)/2.
    m : int
        Size of the sample whose ranks were summed.

    Returns
    -------
    RichResult
        keys: ``U``, ``W``, ``offset``, ``m``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 6.6.
    """
    m = int(m)
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}.")
    W = float(W)
    off = m * (m + 1) / 2.0
    if W < off:
        raise ValueError(f"W = {W} is below the minimum possible rank sum {off}.")
    return RichResult(
        payload={"U": float(W - off), "W": W, "offset": float(off), "m": m,
                 "method": "U = W - m(m+1)/2 (Gibbons Ch. 6.6)"}
    )


def cheatsheet():
    return "gb_binmw: U = W - m(m+1)/2; same test, two bookkeepings"
