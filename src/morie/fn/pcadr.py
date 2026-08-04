# morie.fn -- function file (rootcoder007/morie)
"""Principal component compression of a marker matrix."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['pcadim', 'pca_dimensionality_reduction']


def pcadim(X, k=None):
    """Principal component compression of a marker matrix.

    Formula: Q = X'X/(n-1) on scaled columns; W the eigenvectors of Q; PC = X W; keep the first k columns

    Parameters
    ----------
    X : array-like, shape (n, p)
        One record per row.
    k : int or None
        Number of components retained; None keeps all.

    Returns
    -------
    RichResult
        ``scores``, ``loadings``, ``eigenvalues``, ``compressed``, ``prop_variance``, ``cum_variance``, ``k``, ``n``, ``p``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 2, Sect. 2.8 pp. 63-64.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.  DEFECT FOUND: neither shared core sign-fixes its eigenvectors, so morie.fn._gp_core.pca_compress and the R morie_pca return loadings and scores that differ by a column sign; the three-way parity harness caught it.  The sign is fixed HERE, in the same way morie.fn._tail1core.eigsym does it -- the largest-magnitude entry of every loading column is made positive -- rather than in the shared cores, which this slice must not edit.  With that, a repeated eigenvalue leaves its loadings determined only up to a rotation within the eigenspace and is not a stable quantity in either language.
    """
    out = G.pca_compress(X, k=k)
    Xm = C.mat(X)
    n, p = len(Xm), len(Xm[0])
    kk = p if k is None else int(k)
    if not 1 <= kk <= p:
        raise ValueError("k must lie between 1 and the number of columns of X")
    W = [list(row) for row in out["loadings"]]
    PC = [list(row) for row in out["scores"]]
    # neither shared core sign-fixes its eigenvectors, so the two language
    # arms disagree by a column sign; fix it here with the same rule
    # _tail1core.eigsym uses -- largest-magnitude entry of each column positive.
    for j in range(p):
        r = max(range(p), key=lambda t: abs(W[t][j]))
        if W[r][j] < 0.0:
            for t in range(p):
                W[t][j] = -W[t][j]
            for t in range(n):
                PC[t][j] = -PC[t][j]
    lam = list(out["eigenvalues"])
    return RichResult(payload={
        "scores": PC, "loadings": W, "eigenvalues": lam,
        "compressed": [row[:kk] for row in PC],
        "prop_variance": out["prop_variance"], "cum_variance": out["cum_variance"],
        "k": kk, "n": n, "p": p,
        "method": "PCA compression, MVSML Sect. 2.8"})


pca_dimensionality_reduction = pcadim


def cheatsheet():
    return 'pcadr: Principal component compression of a marker matrix.'
