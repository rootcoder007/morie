# morie.fn -- function file (rootcoder007/morie)
"""MULTISPATI spatial principal component analysis."""

from math import fsum, sqrt

from ._richresult import RichResult
from ._spx import mat, matmul, matvec, sqmat, transpose, topeigs

__all__ = [
    "spatial_pca",
    "spatialpca",
]


def spatial_pca(x, w, naxes=2):
    """MULTISPATI: principal components that maximise spatial autocorrelation.

    NOT IN SCHABENBERGER & GOTWAY -- a fixed-string search for "principal
    component" in the book returns nothing. The method is Dray, S., Said,
    S. & Debias, F. (2008), "Spatial ordination of vegetation data using a
    generalization of Wartenberg's multivariate spatial correlation",
    *Journal of Vegetation Science* 19:45-56 -- named from the general
    literature and NOT verified against a PDF in this corpus.

    Ordinary PCA maximises variance and is blind to the map. MULTISPATI
    diagonalises

        H = (1/n) X' ((W + W')/2) X

    on the centred, unit-variance X, so an axis is scored by its
    Moran-type spatial covariance, not by its variance. The eigenvalues
    can be NEGATIVE, and that is not a bug: a negative axis is a pattern
    of local CONTRAST (neighbouring sites unlike each other), which
    ordinary PCA has no way to express. Eigenvalues are returned signed
    and unsorted-by-magnitude precisely so that structure is visible.

    W is symmetrised before the eigen-decomposition. A row-standardised W
    is ASYMMETRIC, and feeding it to a symmetric eigensolver reads one
    triangle only and silently answers a different question.

    Columns are scaled to unit variance with the 1/n divisor (not 1/(n-1)),
    so that the reported ``total_variance`` is exactly the number of
    columns and the eigenvalues are comparable across data sets.

    Parameters
    ----------
    x : (n, p) array-like
        Site-by-variable matrix.
    w : (n, n) array-like
        Spatial weights, zero diagonal.
    naxes : int
        Number of axes to return, 1 <= naxes <= p.

    Returns
    -------
    RichResult
        ``eigenvalues``, ``loadings``, ``scores``, ``lagged_scores``,
        ``total_variance``, ``n``, ``method``.
    """
    xm = mat(x, "x")
    n = len(xm)
    p = len(xm[0])
    if n < 3:
        raise ValueError("at least 3 sites are needed")
    naxes = int(naxes)
    if naxes < 1 or naxes > p:
        raise ValueError("`naxes` must lie between 1 and the number of "
                         "columns")
    ww = sqmat(w, n, "w")
    for i in range(n):
        if ww[i][i] != 0.0:
            raise ValueError("`w` must have a zero diagonal")

    cols = transpose(xm)
    z = []
    for c in cols:
        m = fsum(c) / n
        d = [t - m for t in c]
        s = sqrt(fsum([t * t for t in d]) / n)
        if s <= 0:
            raise ValueError("a column of `x` is constant and cannot be "
                             "scaled to unit variance")
        z.append([t / s for t in d])
    zz = transpose(z)

    sym = [[0.5 * (ww[i][j] + ww[j][i]) for j in range(n)] for i in range(n)]
    h = matmul(transpose(zz), matmul(sym, zz))
    for i in range(p):
        for j in range(p):
            h[i][j] = h[i][j] / n
    for i in range(p):
        for j in range(i + 1, p):
            av = 0.5 * (h[i][j] + h[j][i])
            h[i][j] = av
            h[j][i] = av

    vals, vecs = topeigs(h, naxes)
    scores = [[fsum([zz[i][j] * vecs[a][j] for j in range(p)])
               for i in range(n)] for a in range(naxes)]
    lagged = [matvec(sym, s) for s in scores]

    return RichResult(payload={
        "eigenvalues": vals,
        "loadings": vecs,
        "scores": scores,
        "lagged_scores": lagged,
        "total_variance": float(p),
        "eigenvalues_may_be_negative": True,
        "weights_symmetrised": True,
        "naxes": float(naxes),
        "n": n,
        "method": ("MULTISPATI spatial PCA (Dray, Said & Debias 2008); "
                   "NOT in Schabenberger & Gotway"),
    })


def cheatsheet():
    return "speptn: MULTISPATI spatial principal component analysis"


# compact alias per ledger/NAMING.md
spatialpca = spatial_pca
