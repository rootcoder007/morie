# morie.fn -- function file (rootcoder007/morie)
"""Block frequencies are uniform over compositions."""

from math import comb

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_block_freq_dist"]


def gibbons_block_freq_dist(m, n, block_counts=None):
    r"""Theorem 2.11.2: when F_X = F_Y, the vector of block
    frequencies -- how many of the n Y-observations fall in each of
    the m + 1 blocks cut by the X order statistics -- is uniform:

    .. math:: P(B_1 = b_1, \dots, B_{m+1} = b_{m+1})
              = \frac{1}{\binom{m+n}{n}}

    for every composition with :math:`\sum b_i = n`. Every placement
    pattern is equally likely, which is why placement statistics are
    distribution-free.

    Parameters
    ----------
    m, n : int
        Sizes of the X (cutting) and Y (placed) samples.
    block_counts : array-like of int, optional
        A specific composition; validated to sum to n.

    Returns
    -------
    RichResult
        keys: ``pmf`` (the common probability), ``n_compositions``
        (C(m+n, n)), ``valid_composition`` (if given), ``m``, ``n``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 2.11.2.
    """
    m, n = int(m), int(n)
    if m < 1 or n < 0:
        raise ValueError("need m >= 1 and n >= 0.")
    total = comb(m + n, n)
    valid = None
    if block_counts is not None:
        b = np.asarray(block_counts, dtype=int).ravel()
        if b.size != m + 1:
            raise ValueError(f"block_counts must have m + 1 = {m + 1} entries.")
        if np.any(b < 0):
            raise ValueError("block counts must be non-negative.")
        valid = bool(b.sum() == n)
    return RichResult(
        payload={
            "pmf": 1.0 / total, "n_compositions": int(total),
            "valid_composition": valid, "m": m, "n": n,
            "method": "Block frequencies uniform over C(m+n, n) compositions",
        }
    )


def cheatsheet():
    return "gb2112: every composition equally likely, p = 1/C(m+n, n)"
