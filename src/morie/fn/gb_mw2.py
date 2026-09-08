# morie.fn -- function file (rootcoder007/morie)
"""Rank-sum and Mann-Whitney computed jointly from data."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["gibbons_mw_rs_equiv"]


def gibbons_mw_rs_equiv(x, y):
    r"""Compute both two-sample forms and demonstrate their identity
    (Gibbons Ch. 6.6): W = sum of x-ranks in the combined sample,
    U = number of (x, y) pairs with x > y (ties counted half), and

    .. math:: U = W - m(m+1)/2

    holds exactly -- returned as a checked boolean, not assumed.

    Parameters
    ----------
    x, y : array-like
        The two samples.

    Returns
    -------
    RichResult
        keys: ``W``, ``U_from_W``, ``U_direct`` (pair count),
        ``identity_holds``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 6.6.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m, n = x.size, y.size
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    ranks = stats.rankdata(np.r_[x, y])
    W = float(ranks[:m].sum())
    U_w = W - m * (m + 1) / 2.0
    gt = np.sum(x[:, None] > y[None, :])
    eq = np.sum(x[:, None] == y[None, :])
    U_d = float(gt + 0.5 * eq)
    return RichResult(
        payload={
            "W": W, "U_from_W": float(U_w), "U_direct": U_d,
            "identity_holds": bool(abs(U_w - U_d) < 1e-9), "m": m, "n": n,
            "method": "W and U computed independently; U = W - m(m+1)/2 checked",
        }
    )


def cheatsheet():
    return "gb_mw2: rank route and pair-count route agree exactly"
