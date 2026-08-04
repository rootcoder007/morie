# morie.fn -- function file (rootcoder007/morie)
"""Principal component analysis in centred log-ratio coordinates."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['clrpca', 'aitchison_clr_pca']


def clrpca(X, k=2):
    """Principal component analysis in centred log-ratio coordinates.

    The compositional analogue of the singular value decomposition: a composition decomposes as the centre perturbed by powered compositions, and the proportion of total variability an r-term approximation retains is the leading eigenvalue share of the centred log-ratio covariance. Eigenvectors are sign-fixed on their largest-magnitude entry, without which the two language arms would disagree on an arbitrary sign.


    Formula: clr(x) = log x - mean(log x); eigendecompose the clr covariance; proportion retained = (s_1^2+...+s_r^2)/sum s^2

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; strictly positive.
    k : int
        Number of components retained.

    Returns
    -------
    RichResult
        ``values``, ``loadings``, ``scores``, ``prop_var``, ``cum_prop``, ``k``, ``n``, ``D``.

    References
    ----------
    Aitchison, A Concise Guide to Compositional Data Analysis,
    Chapter 2.  Verified against the text: the centre estimate is
    xi-hat = C(g_1, ..., g_D) with g_i the geometric mean of the ith
    component, and totvar(x) = trace(Gamma) = (1/D) sum_{i<j}
    var{log(x_i/x_j)}.
    """
    X = C.mat(X)
    n = len(X); D = len(X[0]); k = int(k)
    for row in X:
        if any(v <= 0 for v in row):
            raise ValueError("compositions must be strictly positive")
    L = [[math.log(v) for v in row] for row in X]
    Z = [[L[i][j] - sum(L[i]) / D for j in range(D)] for i in range(n)]
    mu = [sum(Z[i][j] for i in range(n)) / n for j in range(D)]
    Zc = [[Z[i][j] - mu[j] for j in range(D)] for i in range(n)]
    cov = [[sum(Zc[t][i] * Zc[t][j] for t in range(n)) / (n - 1)
            for j in range(D)] for i in range(D)]
    vals, vecs = C.eigsym(cov)
    tot = sum(v for v in vals if v > 0)
    load = [[vecs[i][j] for j in range(k)] for i in range(D)]
    scores = [[sum(Zc[t][i] * vecs[i][j] for i in range(D)) for j in range(k)]
              for t in range(n)]
    prop = [vals[j] / tot for j in range(k)]
    return RichResult(payload={
        "values": vals, "loadings": load, "scores": scores,
        "prop_var": prop, "cum_prop": C.cumsum(prop), "k": k,
        "n": n, "D": D, "method": "Compositional (clr) principal components"})


aitchison_clr_pca = clrpca


def cheatsheet():
    return "aitpca: Principal component analysis in centred log-ratio coordinates."
