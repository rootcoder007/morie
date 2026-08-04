# morie.fn -- function file (rootcoder007/morie)
"""Empirical spatial autocorrelation function R(h) = C(h)/C(0)."""

from math import fsum

from ._richresult import RichResult
from ._spx import eucdist, mat, mean, vec

__all__ = [
    "schabenberger_autocorrelation_function",
    "spacf",
]


def schabenberger_autocorrelation_function(coords, z, bins=None, cutoff=None):
    """Binned empirical autocorrelation function of a spatial random field.

    Schabenberger & Gotway (2005), Sec. 1.4.2, define the covariance
    function ``C(s, h) = Cov[Z(s), Z(s+h)]`` and the correlogram

        R(s, h) = C(h) / sqrt(Var[Z(s)] Var[Z(s+h)]),

    which under second-order stationarity (same section, p. 27) collapses
    to ``R(h) = C(h)/sigma^2``; Chapter problem 1.14 writes the same
    quantity as ``R(h) = C(h)/C(0)`` and is where the "correlogram" name
    is attached to it.

    ``C(0)`` is estimated by ``n^-1 sum (Z(s_i) - Zbar)^2`` and, for a lag
    class ``b``, ``C(h_b)`` by the mean of ``(Z(s_i) - Zbar)(Z(s_j) - Zbar)``
    over the pairs whose separation falls in ``b``. Pairs are counted once,
    i < j.

    Note the book's own warning at Sec. 2.2: qualitatively different
    processes can share a correlogram, so ``R(h)`` is an INCOMPLETE
    description of the second-order structure. It is reported here beside
    the raw covariances, not instead of them.

    Parameters
    ----------
    coords : (n, d) array-like
        Site coordinates.
    z : (n,) array-like
        Attribute values.
    bins : int or sequence of float, optional
        Number of equal-width lag classes (default 10), or explicit upper
        edges.
    cutoff : float, optional
        Largest separation used; defaults to the maximum pair distance.

    Returns
    -------
    RichResult
        ``lags`` (upper edges), ``centres``, ``cov``, ``acf``, ``c0``,
        ``npairs``, ``n``, ``method``.
    """
    zz = vec(z, "z")
    cc = mat(coords, "coords")
    n = len(zz)
    if len(cc) != n:
        raise ValueError("`coords` has %d rows but `z` has %d values"
                         % (len(cc), n))
    if n < 3:
        raise ValueError("at least 3 sites are needed for a lag class")
    d = [t - mean(zz) for t in zz]
    c0 = fsum([t * t for t in d]) / n
    if c0 <= 0:
        raise ValueError("`z` is constant; C(0) is zero and R(h) undefined")

    h = []
    prod = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            h.append(eucdist(cc[i], cc[j]))
            prod.append(d[i] * d[j])
    hmax = max(h)
    if cutoff is not None:
        hmax = float(cutoff)
        if hmax <= 0:
            raise ValueError("`cutoff` must be positive")
    if bins is None:
        edges = [hmax * (k + 1) / 10.0 for k in range(10)]
    elif isinstance(bins, int):
        if bins < 1:
            raise ValueError("`bins` must be at least 1")
        edges = [hmax * (k + 1) / float(bins) for k in range(bins)]
    else:
        edges = vec(bins, "bins")
        if any(edges[k] <= edges[k - 1] for k in range(1, len(edges))):
            raise ValueError("`bins` edges must increase")

    lo = 0.0
    cov = []
    acf = []
    npairs = []
    centres = []
    for e in edges:
        acc = [prod[k] for k in range(len(h)) if lo < h[k] <= e]
        npairs.append(len(acc))
        centres.append(0.5 * (lo + e))
        if acc:
            c = fsum(acc) / len(acc)
            cov.append(c)
            acf.append(c / c0)
        else:
            cov.append(float("nan"))
            acf.append(float("nan"))
        lo = e

    return RichResult(payload={
        "lags": edges,
        "centres": centres,
        "cov": cov,
        "acf": acf,
        "c0": c0,
        "npairs": npairs,
        "n": n,
        "incomplete_description_of_second_order_structure": True,
        "method": ("Empirical correlogram R(h)=C(h)/C(0); Schabenberger & "
                   "Gotway (2005) Sec. 1.4.2 and Chapter problem 1.14"),
    })


def cheatsheet():
    return "spackf: empirical spatial correlogram R(h)=C(h)/C(0)"


# compact alias per ledger/NAMING.md
spacf = schabenberger_autocorrelation_function
