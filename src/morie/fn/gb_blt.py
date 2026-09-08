# morie.fn -- function file (rootcoder007/morie)
"""Concordance for balanced incomplete block rankings."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["gibbons_balance_incomplete"]


def gibbons_balance_incomplete(rankings, lam=None, n=None, k=None):
    r"""Coefficient of concordance for a balanced incomplete block
    (BIB) design.

    In a BIB design each of the b judges ranks exactly m of the n
    objects, every object is ranked r times, and every *pair* of
    objects appears together in exactly :math:`\lambda` rankings.
    The concordance statistic (Gibbons Ch. 12.5) is

    .. math:: W_b = \frac{12 S_b}{\lambda^2 n (n^2 - 1)}
              \cdot \frac{(m - 1)^2 \lambda^2}{r^2 (m^2-1)\lambda^2 /
              [\;\cdot\;]}

    reduced here to the operative form
    :math:`W_b = 12 S_b / [\lambda^2 n (n^3 - n) / (m - 1)]`-style
    normalisation via the design identities
    :math:`r(m - 1) = \lambda(n - 1)` and :math:`bm = nr`; those two
    identities are CHECKED against the supplied rankings, and a
    design that fails them raises -- an unbalanced layout silently
    scored with the BIB constant would be wrong by design, not by
    rounding.

    Parameters
    ----------
    rankings : array-like, shape (b, n)
        NaN for objects a judge did not rank; non-NaN entries are
        ranks 1..m within the block.
    lam : int, optional
        Pair-concurrence count; inferred from the design if omitted.
    n, k : int, optional
        Accepted for interface compatibility (n objects, k = b
        judges); shapes are taken from ``rankings``.

    Returns
    -------
    RichResult
        keys: ``W_b``, ``S_b``, ``lambda_``, ``r_per_object``,
        ``m_per_block``, ``b``, ``n``, ``chi2``, ``df``, ``p_value``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 12.5.

    Durbin, J. (1951). Incomplete blocks in ranking experiments.
    *British Journal of Statistical Psychology*, 4(2), 85-90.
    """
    R = np.asarray(rankings, dtype=float)
    if R.ndim != 2:
        raise ValueError("rankings must be 2-D (b blocks x n objects).")
    b, nn = R.shape
    seen = ~np.isnan(R)
    m_per = seen.sum(axis=1)
    if np.unique(m_per).size != 1:
        raise ValueError("not a BIB design: blocks rank different numbers of objects.")
    m = int(m_per[0])
    if m < 2:
        raise ValueError("each block must rank at least 2 objects.")
    r_per = seen.sum(axis=0)
    if np.unique(r_per).size != 1:
        raise ValueError("not a BIB design: objects are ranked unequally often.")
    r = int(r_per[0])
    # pair concurrence
    conc = seen.T.astype(int) @ seen.astype(int)
    off = conc[np.triu_indices(nn, 1)]
    if np.unique(off).size != 1:
        raise ValueError("not a BIB design: pair concurrence is not constant.")
    lam_obs = int(off[0])
    if lam is not None and int(lam) != lam_obs:
        raise ValueError(f"supplied lambda = {lam} but the design has {lam_obs}.")
    lam = lam_obs
    # design identity r(m - 1) = lambda(n - 1)
    if r * (m - 1) != lam * (nn - 1):
        raise ValueError("design identities fail; not a valid BIB layout.")

    col = np.nansum(R, axis=0)
    S_b = float(np.sum((col - col.mean()) ** 2))
    # Durbin's normalisation: the maximum attainable S_b under perfect
    # agreement in a BIB design is lambda^2 n (n^2 - 1)(m + 1)/(12(m - 1))
    denom = lam**2 * nn * (nn**2 - 1) * (m + 1) / (12.0 * (m - 1))
    W_b = S_b / denom
    # Durbin's chi-square statistic for the same design
    chi2 = 12.0 * (nn - 1) / (r * nn * (m**2 - 1)) * float(
        np.sum((col - r * (m + 1) / 2.0) ** 2)
    )
    return RichResult(
        payload={
            "W_b": float(min(W_b, 1.0)), "S_b": S_b, "lambda_": lam,
            "r_per_object": r, "m_per_block": m, "b": int(b), "n": int(nn),
            "chi2": float(chi2), "df": int(nn - 1),
            "p_value": float(stats.chi2.sf(chi2, nn - 1)),
            "method": "BIB concordance with verified design identities (Gibbons Ch. 12.5)",
        }
    )


def cheatsheet():
    return "gb_blt: BIB W_b; design identities r(m-1) = lam(n-1) are checked, not assumed"
