# morie.fn -- function file (rootcoder007/morie)
"""McNemar's test for symmetry in paired binary data."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['mcnemarq', 'gibbons_mcnemar']


def mcnemarq(table, correct=False):
    """McNemar's chi-square on the discordant pairs, eq. (14.5.1).

    Book p. 523.  For paired binary data the null hypothesis
    H0: theta_{1.} = theta_{.1} concerns only the two discordant
    cells, and

    .. math:: Q = \\frac{(X_{12} - X_{21})^2}{X_{12} + X_{21}}

    is approximately chi-square with 1 degree of freedom.  The book
    warns explicitly about the accuracy of that approximation when the
    expected discordant counts are small, so the exact binomial test
    conditional on X12 + X21 is returned alongside as ``p_exact``.

    Parameters
    ----------
    table : sequence of sequence of float
        The 2 x 2 table of paired counts.
    correct : bool, optional
        Apply a continuity correction of 1 to |X12 - X21| (default
        False).

    Returns
    -------
    RichResult
        keys ``statistic``, ``df``, ``p_value``, ``p_exact``,
        ``x12``, ``x21``, ``ndisc``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 14.5, eq. (14.5.1), p. 523.
    """
    tb = [[float(v) for v in row] for row in table]
    if len(tb) != 2 or any(len(row) != 2 for row in tb):
        raise ValueError("table must be 2 x 2.")
    x12 = tb[0][1]
    x21 = tb[1][0]
    nd = x12 + x21
    if nd <= 0:
        raise ValueError("there are no discordant pairs.")
    d = abs(x12 - x21)
    if correct:
        d = max(0.0, d - 1.0)
    q = d * d / nd
    k = int(round(min(x12, x21)))
    ni = int(round(nd))
    pex = min(
        1.0,
        2.0
        * sum(math.comb(ni, i) for i in range(k + 1))
        * 0.5**ni,
    )
    return RichResult(
        payload={
            "statistic": float(q),
            "df": 1,
            "p_value": float(stats.chi2.sf(q, 1)),
            "p_exact": float(pex),
            "x12": float(x12),
            "x21": float(x21),
            "ndisc": float(nd),
            "method": "McNemar test, eq. (14.5.1)",
        }
    )


gibbons_mcnemar = mcnemarq
