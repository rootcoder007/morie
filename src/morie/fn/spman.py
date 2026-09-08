# morie.fn -- function file (rootcoder007/morie)
"""Mantel's clustering statistics M1 and M2, eqs (1.4) and (1.5)."""

from math import fsum

from ._richresult import RichResult
from ._spx import eucdist, mat, mean, sqmat, vec

__all__ = [
    "schabenberger_mantel_test",
    "mantelm2",
]


def schabenberger_mantel_test(coords, x, w=None, u=None):
    """Mantel's statistics for clustering, Schabenberger & Gotway Sec. 1.3.1.

    The book, after Mantel (1967), forms a spatial proximity matrix W and
    an attribute proximity matrix U with zero diagonals and reports

        M1 = sum_{i=1}^{n-1} sum_{j=i+1}^{n} W_ij U_ij          eq (1.4)
        M2 = sum_{i=1}^{n}   sum_{j=1}^{n}   W_ij U_ij          eq (1.5)

    so M1 runs over the n(n-1)/2 unique pairs and M2 over all n(n-1)
    ordered pairs. The same section reads the pair (U_ij, W_ij) as a
    regression through the origin, ``U_ij = beta W_ij + e_ij``, whose slope
    estimator is

        beta = M2 / sum_i sum_j W_ij^2,

    and that slope is returned as ``beta``: negative slope = positive
    spatial autocorrelation, because small distances then pair with small
    attribute differences.

    Defaults follow the book's own worked choices in Sec. 1.3.1:
    ``W_ij = ||s_i - s_j||`` and ``U_ij = |Z(s_i) - Z(s_j)|``.

    No p-value is returned here. The book lists four routes to one
    (permutation, Monte Carlo, and two asymptotic Z-tests) and only the
    Gaussian Z-test is closed-form; it lives in
    ``schabenberger_mantel_standard``. Reporting a rank correlation's
    p-value under this name, which is what the generated stub did, tests a
    different hypothesis entirely.

    Parameters
    ----------
    coords : (n, d) array-like
        Site coordinates; ignored when `w` is supplied.
    x : (n,) array-like
        Attribute values; ignored when `u` is supplied.
    w : (n, n) array-like, optional
        Explicit spatial proximity matrix.
    u : (n, n) array-like, optional
        Explicit attribute proximity matrix.

    Returns
    -------
    RichResult
        ``m1``, ``m2``, ``beta``, ``sw2``, ``s0``, ``n``, ``method``.
    """
    z = vec(x, "x")
    n = len(z)
    if n < 2:
        raise ValueError("at least 2 sites are needed for a pair")
    if w is None:
        cc = mat(coords, "coords")
        if len(cc) != n:
            raise ValueError("`coords` has %d rows but `x` has %d values"
                             % (len(cc), n))
        ww = [[eucdist(cc[i], cc[j]) if i != j else 0.0 for j in range(n)]
              for i in range(n)]
    else:
        ww = sqmat(w, n, "w")
        for i in range(n):
            if ww[i][i] != 0.0:
                raise ValueError("`w` must have a zero diagonal (W_ii = 0)")
    if u is None:
        uu = [[abs(z[i] - z[j]) if i != j else 0.0 for j in range(n)]
              for i in range(n)]
    else:
        uu = sqmat(u, n, "u")
        for i in range(n):
            if uu[i][i] != 0.0:
                raise ValueError("`u` must have a zero diagonal (U_ii = 0)")

    m1 = fsum([ww[i][j] * uu[i][j] for i in range(n - 1)
               for j in range(i + 1, n)])
    m2 = fsum([ww[i][j] * uu[i][j] for i in range(n) for j in range(n)])
    sw2 = fsum([ww[i][j] * ww[i][j] for i in range(n) for j in range(n)])
    if sw2 <= 0:
        raise ValueError("all spatial proximities are zero; beta undefined")

    return RichResult(payload={
        "m1": m1,
        "m2": m2,
        "beta": m2 / sw2,
        "sw2": sw2,
        "s0": fsum([fsum(row) for row in ww]),
        "mean_attribute": mean(z),
        "n": n,
        "method": ("Mantel statistics M1 and M2, Schabenberger & Gotway "
                   "(2005) eqs (1.4)-(1.5), Sec. 1.3.1; Mantel (1967)"),
    })


def cheatsheet():
    return "spman: Mantel M1/M2 clustering statistics, eqs (1.4)-(1.5)"


# compact alias per ledger/NAMING.md
mantelm2 = schabenberger_mantel_test
