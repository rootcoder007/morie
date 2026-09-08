# morie.fn -- function file (rootcoder007/morie)
"""Chi-square approximation to the Kruskal-Wallis statistic."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['kwchi', 'gibbons_kw_chi2_approx']


def kwchi(h, k, ns=None):
    """Reference chi-square tail for H, with the book's caveat.

    Section 10.4.1 (book p. 357): H is asymptotically chi-square on
    k - 1 degrees of freedom, and the approximation "is generally
    satisfactory except when k = 3 and the sample sizes are five or
    less" -- exactly the case Table K covers.  ``table_k`` flags that
    regime so the caller knows the exact table should be used instead.

    Parameters
    ----------
    h : float
        Observed H.
    k : int
        Number of samples, k >= 2.
    ns : sequence of int, optional
        The sample sizes, used only to set the ``table_k`` flag.

    Returns
    -------
    RichResult
        keys ``statistic``, ``df``, ``p_value``, ``table_k`` (1 when
        the exact table is preferable), ``k``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 10.4.1, p. 357; Table K, p. 582.
    """
    h = float(h)
    k = int(k)
    if k < 2:
        raise ValueError("k must be at least 2.")
    df = k - 1
    flag = 0
    if ns is not None:
        nv = [int(v) for v in ns]
        if k == 3 and all(v <= 5 for v in nv):
            flag = 1
    return RichResult(
        payload={
            "statistic": h,
            "df": int(df),
            "p_value": float(stats.chi2.sf(h, df)),
            "table_k": int(flag),
            "k": k,
            "method": "chi-square approximation to H (Sec. 10.4.1)",
        }
    )


gibbons_kw_chi2_approx = kwchi
