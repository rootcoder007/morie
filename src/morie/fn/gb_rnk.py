# morie.fn -- function file (rootcoder007/morie)
"""Rank as a function of the EDF."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_rank_def"]


def gibbons_rank_def(x):
    r"""Section 2.11.3: the rank of X_i in a tie-free sample is

    .. math:: \mathrm{rank}(X_i) = \sum_{j=1}^n I(X_j \le X_i)
              = n\, S_n(X_i),

    the EDF evaluated at the point, scaled by n. Ties are rejected
    rather than midranked here because the identity as stated
    requires distinct values; midranking is a convention layered on
    top, not part of the definition.

    Parameters
    ----------
    x : array-like
        Sample of distinct values.

    Returns
    -------
    RichResult
        keys: ``ranks``, ``edf_values`` (ranks / n), ``n``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 2.11.3.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 1:
        raise ValueError("x must be non-empty.")
    if np.unique(x).size != n:
        raise ValueError("the rank identity requires distinct values (no ties).")
    ranks = np.sum(x[None, :] <= x[:, None], axis=1)
    return RichResult(
        payload={
            "ranks": ranks.astype(int), "edf_values": ranks / n, "n": int(n),
            "method": "rank(X_i) = n S_n(X_i) (Gibbons Ch. 2.11.3)",
        }
    )


def cheatsheet():
    return "gb_rnk: rank = n * EDF at the point; needs distinct values"
