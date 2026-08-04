# morie.fn -- function file (rootcoder007/morie)
"""VanRaden Method 1 genomic relationship matrix.

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

__all__ = ["vanraden_method1"]


def _allele_freq(M, freq):
    """Minor allele frequencies p_j.  For markers coded 0/1/2 the
    frequency is the column mean over 2 (MVSML p.51, phat =
    colMeans(X)/2)."""
    if freq is not None:
        return [float(v) for v in freq]
    n = len(M)
    return [sum(row[j] for row in M) / (2.0 * n) for j in range(len(M[0]))]


def vanraden_method1(marker_matrix, freq=None):
    """Centred genomic relationship matrix,

        G = Z Z' / (2 sum_j p_j (1 - p_j)),   Z_ij = M_ij - 2 p_j.

    Each marker column is centred by twice its allele frequency, which
    is its expectation under Hardy-Weinberg, and the whole matrix is
    divided by the summed marker variances so that G is on the scale
    of the numerator relationship matrix.  This is VanRaden's Method 1
    and the MVSML (2022) Method 2 of p.51.

    Parameters
    ----------
    marker_matrix : (J, p) array-like coded 0, 1, 2 (lines by markers).
    freq : optional (p,) allele frequencies; column means over 2 by
        default.

    Returns
    -------
    RichResult with keys estimate (the mean diagonal of G), G, freq,
    denominator, n_lines, n_markers, method.

    References
    ----------
    VanRaden (2008) J Dairy Sci 91:4414-4423, Method 1; MVSML (2022)
    sec. 2.4 p.51 Method 2.
    """
    M = [[float(v) for v in row] for row in marker_matrix]
    J = len(M)
    p = len(M[0])
    pj = _allele_freq(M, freq)
    Z = [[M[i][j] - 2.0 * pj[j] for j in range(p)] for i in range(J)]
    den = 2.0 * sum(q * (1.0 - q) for q in pj)
    G = [[sum(Z[i][k] * Z[j][k] for k in range(p)) / den
          for j in range(J)] for i in range(J)]
    return with_describe_pointer(RichResult(payload={
        "estimate": float(sum(G[i][i] for i in range(J)) / J),
        "G": G, "freq": pj, "denominator": float(den),
        "n_lines": J, "n_markers": p,
        "method": "VanRaden (2008) method 1 genomic relationship matrix",
    }), "vanr1")


def cheatsheet():
    return "vanr1: VanRaden Method 1 genomic relationship matrix"


# compact alias per ledger/NAMING.md
vanraden1 = vanraden_method1
