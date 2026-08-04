# morie.fn -- function file (rootcoder007/morie)
"""Yang et al. realized (unified additive) genomic relationship matrix.

Yang, J. et al. (2010).  Common SNPs explain a large proportion of the
heritability for human height.  Nat Genet 42:565-569, and the GCTA
software the same group maintains.  The GCTA documentation was read
directly (yanglab.westlake.edu.cn/software/gcta, "Making a GRM") and
cross-checked against the snpReady::G.matrix "UAR" documentation.
"""

from ._richresult import RichResult, with_describe_pointer

__all__ = ["yang_realized_relationship"]


def yang_realized_relationship(marker_matrix, freq=None, yang_diagonal=False):
    """Realized relationship matrix scaled marker by marker,

        A_jk = (1/N) sum_i (x_ij - 2 p_i)(x_ik - 2 p_i)
                     / (2 p_i (1 - p_i)),

    where N is the number of markers.  Unlike VanRaden Method 1, which
    applies one overall divisor, each locus is standardized by its own
    Hardy-Weinberg variance before the average is taken.

    The original paper gives the diagonal a different expression,

        A_jj = 1 + (1/N) sum_i [x_ij^2 - (1 + 2 p_i) x_ij + 2 p_i^2]
                          / (2 p_i (1 - p_i)),

    while GCTA later changed the diagonal "to be the same as that for
    the off-diagonal elements".  ``yang_diagonal=True`` selects the
    1-based paper form; the default follows current GCTA and the
    formula the stub this module replaces carried.

    Parameters
    ----------
    marker_matrix : (J, p) array-like coded 0, 1, 2.
    freq : optional (p,) allele frequencies; column means over 2 by
        default.
    yang_diagonal : bool, use the Yang et al. (2010) diagonal.

    Returns
    -------
    RichResult with keys estimate (the mean diagonal of A), A, freq,
    n_lines, n_markers, yang_diagonal, method.

    References
    ----------
    Yang et al. (2010) Nat Genet 42:565-569; GCTA documentation.
    """
    M = [[float(v) for v in row] for row in marker_matrix]
    J = len(M)
    p = len(M[0])
    if freq is not None:
        pi = [float(v) for v in freq]
    else:
        pi = [sum(row[j] for row in M) / (2.0 * J) for j in range(p)]
    var = [2.0 * q * (1.0 - q) for q in pi]
    A = []
    for i in range(J):
        row = []
        for j in range(J):
            if i == j and yang_diagonal:
                s = 0.0
                for k in range(p):
                    if var[k] <= 0:
                        continue
                    x = M[i][k]
                    s += (x * x - (1.0 + 2.0 * pi[k]) * x
                          + 2.0 * pi[k] * pi[k]) / var[k]
                row.append(1.0 + s / p)
            else:
                s = 0.0
                for k in range(p):
                    if var[k] <= 0:
                        continue
                    s += ((M[i][k] - 2.0 * pi[k])
                          * (M[j][k] - 2.0 * pi[k]) / var[k])
                row.append(s / p)
        A.append(row)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(sum(A[i][i] for i in range(J)) / J),
        "A": A, "freq": pi, "n_lines": J, "n_markers": p,
        "yang_diagonal": bool(yang_diagonal),
        "method": "Yang et al. (2010) realized relationship matrix",
    }), "yangr")


def cheatsheet():
    return "yangr: Yang et al. realized genomic relationship matrix"


# compact alias per ledger/NAMING.md
yangrel = yang_realized_relationship
