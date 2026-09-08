# morie.fn -- function file (rootcoder007/morie)
"""Local indicators of spatial association, eq (1.17)."""

from math import fsum

from ._richresult import RichResult
from ._spx import mean, sqmat, vec

__all__ = [
    "schabenberger_lisa",
    "lisai",
]


def schabenberger_lisa(x, w):
    """Anselin's local Moran's I, Schabenberger & Gotway eq (1.17).

    Sec. 1.3.3 ("Localized Indicators of Spatial Autocorrelation") starts
    from Mantel's M2 of Sec. 1.3.1, writes it as a sum of per-site
    contributions, and gives the local Moran statistic

        I(s_i) = n / {(n-1) S^2} (Z(s_i) - Zbar) sum_j w_ij (Z(s_j) - Zbar)

    which satisfies ``sum_i I(s_i) = w.. I`` with ``I`` the global Moran
    statistic of eq (1.14). That identity is checked here and returned as
    ``sum_identity_gap``; it is zero to rounding, and a non-zero value
    means the weights or the scaling are wrong.

    The randomization mean is the book's

        E_r[I(s_i)] = -(n-1)^-1 sum_j w_ij.

    Because ``(n-1) S^2 = sum_i (Z(s_i) - Zbar)^2``, the leading constant
    is simply ``n / sum_i (Z(s_i) - Zbar)^2``; the S^2 spelling is kept in
    the docstring because that is how the book writes it.

    Parameters
    ----------
    x : (n,) array-like
        Attribute values.
    w : (n, n) array-like
        Spatial connectivity weights, zero diagonal.

    Returns
    -------
    RichResult
        ``local``, ``expectation``, ``lagged``, ``global_i``, ``s0``,
        ``sum_identity_gap``, ``n``, ``method``.
    """
    z = vec(x, "x")
    n = len(z)
    if n < 3:
        raise ValueError("at least 3 sites are needed")
    ww = sqmat(w, n, "w")
    for i in range(n):
        if ww[i][i] != 0.0:
            raise ValueError("`w` must have a zero diagonal; "
                             "a site is not its own neighbour")
    m = mean(z)
    d = [t - m for t in z]
    ss = fsum([t * t for t in d])
    if ss <= 0:
        raise ValueError("`x` is constant; local Moran's I is undefined")
    s0 = fsum([fsum(row) for row in ww])
    if s0 <= 0:
        raise ValueError("total weight w.. must be positive")

    lagged = [fsum([ww[i][j] * d[j] for j in range(n)]) for i in range(n)]
    local = [n * d[i] * lagged[i] / ss for i in range(n)]
    expect = [-fsum(ww[i]) / (n - 1.0) for i in range(n)]
    gi = n * fsum([ww[i][j] * d[i] * d[j] for i in range(n)
                   for j in range(n)]) / (s0 * ss)
    gap = fsum(local) - s0 * gi

    return RichResult(payload={
        "local": local,
        "expectation": expect,
        "lagged": lagged,
        "global_i": gi,
        "s0": s0,
        "sum_identity_gap": gap,
        "n": n,
        "method": ("Local Moran's I, Schabenberger & Gotway (2005) "
                   "eq (1.17), Sec. 1.3.3; after Anselin (1995)"),
    })


def cheatsheet():
    return "splisa: local Moran's I, eq (1.17)"


# compact alias per ledger/NAMING.md
lisai = schabenberger_lisa
