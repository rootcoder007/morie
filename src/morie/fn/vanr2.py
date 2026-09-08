# morie.fn -- function file (rootcoder007/morie)
"""VanRaden Method 2 genomic relationship matrix (marker weighted).

Sources.  VanRaden, P.M. (2008).  Efficient methods to compute genomic
predictions.  J Dairy Sci 91:4414-4423 -- the journal full text is
behind a 403 for both the publisher and ScienceDirect mirrors, so the
method definitions were taken from the MVSML (2022) chapter-2 split
PDF sec. 2.4 pp.50-52, which reproduces them, cross-checked against
the snpReady::G.matrix reference documentation.

Numbering warning, verified by reading both.  The book renumbers
VanRaden: the book's Method 1 is G = XX'/p (an uncentred inner
product), its Method 2 is the centred G = ZZ'/(2 sum_j p_j(1-p_j)),
and its Method 3 divides each column by its sample standard
deviation.  VanRaden's own paper calls the centred form Method 1 and
the per-marker weighted form Method 2.  The stub docstrings this
module replaces carried VanRaden's paper numbering, so the paper
numbering is what is implemented, and the book equivalents are named
in each docstring.
"""

from ._richresult import RichResult, with_describe_pointer

__all__ = ["vanraden_method2"]


def _allele_freq(M, freq):
    if freq is not None:
        return [float(v) for v in freq]
    n = len(M)
    return [sum(row[j] for row in M) / (2.0 * n) for j in range(len(M[0]))]


def vanraden_method2(marker_matrix, weights=None, freq=None):
    """Weighted genomic relationship matrix,

        G = sum_j w_j z_j z_j' / (2 sum_j w_j p_j (1 - p_j)),

    with z_j the jth centred marker column, M_ij - 2 p_j.  VanRaden's
    Method 2 weights each locus by the reciprocal of its expected
    variance, w_j = 1 / (2 p_j (1 - p_j)), which is the default here;
    that gives rare markers more influence than Method 1 does, where
    every locus carries weight one.  Passing ``weights`` of all ones
    reproduces Method 1 exactly.

    Parameters
    ----------
    marker_matrix : (J, p) array-like coded 0, 1, 2.
    weights : optional (p,) per-marker weights; VanRaden's
        1 / (2 p_j (1 - p_j)) by default.  A marker that is fixed
        (p_j = 0 or 1) has zero variance and is given weight zero.
    freq : optional (p,) allele frequencies.

    Returns
    -------
    RichResult with keys estimate (the mean diagonal of G), G, freq,
    weights, denominator, n_lines, n_markers, method.

    References
    ----------
    VanRaden (2008) J Dairy Sci 91:4414-4423, Method 2.
    """
    M = [[float(v) for v in row] for row in marker_matrix]
    J = len(M)
    p = len(M[0])
    pj = _allele_freq(M, freq)
    var = [2.0 * q * (1.0 - q) for q in pj]
    if weights is None:
        w = [1.0 / v if v > 0 else 0.0 for v in var]
    else:
        w = [float(v) for v in weights]
    Z = [[M[i][j] - 2.0 * pj[j] for j in range(p)] for i in range(J)]
    den = sum(w[j] * var[j] for j in range(p))
    G = [[sum(w[k] * Z[i][k] * Z[j][k] for k in range(p)) / den
          for j in range(J)] for i in range(J)]
    return with_describe_pointer(RichResult(payload={
        "estimate": float(sum(G[i][i] for i in range(J)) / J),
        "G": G, "freq": pj, "weights": w, "denominator": float(den),
        "n_lines": J, "n_markers": p,
        "method": "VanRaden (2008) method 2 weighted relationship matrix",
    }), "vanr2")


def cheatsheet():
    return "vanr2: VanRaden Method 2 genomic relationship matrix (weighted)"


# compact alias per ledger/NAMING.md
vanraden2 = vanraden_method2
