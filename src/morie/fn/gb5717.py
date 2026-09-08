# morie.fn -- function file (rootcoder007/morie)
"""Signed-rank test used as a test of symmetry about a known centre."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['wsrsym', 'gibbons_wsrt_symmetry']


def wsrsym(x, centre=0.0):
    """Test H0: the population is symmetric about ``centre``.

    Section 5.7.7 (book p. 211).  The signed-rank statistic assumes
    symmetry as well as a median location, so with the centre held
    fixed and known, rejection is evidence against symmetry itself.
    The statistic and its standardisation are those of Sec. 5.7; the
    skewness of the differences is returned as a direction indicator.

    Parameters
    ----------
    x : sequence of float
        Sample, n >= 2.
    centre : float, optional
        Hypothesised centre of symmetry (default 0).

    Returns
    -------
    RichResult
        keys ``statistic`` (T+), ``z``, ``p_value``, ``mean``,
        ``var``, ``skewdir`` (+1 if T+ exceeds its null mean),
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.7.7, p. 211.
    """
    ds = [float(v) - float(centre) for v in x]
    ds = [v for v in ds if v != 0.0]
    n = len(ds)
    if n < 2:
        raise ValueError("need at least 2 non-zero differences.")
    a = [abs(v) for v in ds]
    order = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    tplus = sum(ranks[i] for i in range(n) if ds[i] > 0.0)
    mean = n * (n + 1.0) / 4.0
    var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
    z = (tplus - mean) / math.sqrt(var)
    return RichResult(
        payload={
            "statistic": float(tplus),
            "z": float(z),
            "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
            "mean": float(mean),
            "var": float(var),
            "skewdir": 1 if tplus > mean else (-1 if tplus < mean else 0),
            "n": int(n),
            "method": "signed-rank test of symmetry about a known centre",
        }
    )


gibbons_wsrt_symmetry = wsrsym
