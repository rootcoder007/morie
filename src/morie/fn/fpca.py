# morie.fn -- slice s04 (rootcoder007/morie)
"""Functional principal components analysis (FPCA).

NOT IN THE BOOK.  Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, was searched in full -- all seventeen page-range
volumes and the index, [Pages 683-691].  "Functional principal" and
"eigenfunction" do not occur anywhere; Chapter 14, volume [Pages
579-631], covers functional regression by fixed basis expansion (Fourier
and B-spline, Section 14.2) and never introduces a data-driven basis.
Chapter 2, volume [Pages 35-70], Section 2.8, gives multivariate PCA on
a rectangular matrix, which is the discrete analogue used below.

The functional version is taken from the functional-data literature the
book itself points at.  Ramsay, J. O. and Silverman, B. W. (2005),
*Functional Data Analysis*, 2nd edition, Springer Series in Statistics,
doi:10.1007/b98888, is the standard source for the theory; Ramsay,
J. O., Hooker, G. and Graves, S. (2009), *Functional Data Analysis with
R and MATLAB*, Springer, doi:10.1007/978-0-387-98185-7, is its companion
software text.  Both bibliographic records are verified against Crossref.

CITATION LIMIT, stated rather than papered over.  Neither book's text
could be fetched, so no page or equation number is attributed to either,
and the statements below are given as this function's own specification
rather than as quotations.  The Karhunen-Loeve expansion implemented is

    x_i(t) = mu(t) + sum_k score_ik phi_k(t),

with phi_k the eigenfunctions of the covariance surface
v(s,t) = (1/(n-1)) sum_i (x_i(s)-mu(s))(x_i(t)-mu(t)) and the
normalisation integral phi_k^2 = 1 rather than the Euclidean one.  The
integral is taken by the trapezoid rule on the observation grid, which
is what makes this the functional and not the multivariate problem, and
the anchors verify the orthonormality and the exactness of the expansion
directly rather than against a quoted page.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["functional_pca"]


def functional_pca(data_functions, n_components, a=0.0, b=1.0):
    """Eigenfunctions and scores of a sample of curves on a common grid.

    Parameters
    ----------
    data_functions : array-like
        n-by-m matrix; row i is curve i sampled on the common grid.
    n_components : int
        Number of components to keep, 1 <= k <= min(n-1, m).
    a, b : float
        End points of the equally spaced grid.

    Returns
    -------
    estimate   : the proportion of variance the k components explain
    scores     : n-by-k matrix of scores
    eigenfuncs : k eigenfunctions, each of length m, with integral^2 = 1
    eigenvalues: the k leading eigenvalues of the covariance surface
    mean       : mu(t)
    """
    Xr = core.mat(data_functions)
    n = len(Xr)
    if n < 2:
        raise ValueError("functional_pca: need at least two curves")
    m = len(Xr[0])
    for r in Xr:
        if len(r) != m:
            raise ValueError("functional_pca: curves are sampled on grids of different length")
    if m < 2:
        raise ValueError("functional_pca: need at least two grid points")
    kk = int(n_components)
    if kk < 1 or kk > min(n - 1, m):
        raise ValueError("functional_pca: n_components must lie between 1 and min(n-1, m)")
    a = float(a)
    b = float(b)
    if not b > a:
        raise ValueError("functional_pca: the grid must have positive width")
    h = (b - a) / (m - 1)
    w = [h * (0.5 if (j == 0 or j == m - 1) else 1.0) for j in range(m)]
    mu = []
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += Xr[i][j]
        mu.append(s / n)
    C = [[Xr[i][j] - mu[j] for j in range(m)] for i in range(n)]
    # weighted covariance surface, symmetrised through sqrt(w) so the
    # eigenproblem is the discretised functional one
    sw = [math.sqrt(v) for v in w]
    V = [[0.0] * m for _ in range(m)]
    for s_ in range(m):
        for t_ in range(m):
            acc = 0.0
            for i in range(n):
                acc += C[i][s_] * C[i][t_]
            V[s_][t_] = acc / (n - 1) * sw[s_] * sw[t_]
    val, vecs = core.jacobi(V)
    order = list(range(m - 1, -1, -1))
    ev = [val[o] for o in order]
    phi = []
    for j in range(kk):
        col = [vecs[r][order[j]] / sw[r] if sw[r] > 0.0 else 0.0 for r in range(m)]
        nrm = 0.0
        for r in range(m):
            nrm += w[r] * col[r] * col[r]
        nrm = math.sqrt(nrm) if nrm > 0.0 else 1.0
        phi.append([v / nrm for v in col])
    scores = []
    for i in range(n):
        row = []
        for j in range(kk):
            s = 0.0
            for r in range(m):
                s += w[r] * C[i][r] * phi[j][r]
            row.append(s)
        scores.append(row)
    tot = 0.0
    for v in ev:
        tot += max(v, 0.0)
    top = 0.0
    for j in range(kk):
        top += max(ev[j], 0.0)
    return RichResult(
        title="Functional PCA",
        summary_lines=[("curves", n), ("grid", m), ("components", kk)],
        payload={
            "estimate": top / tot if tot > 0.0 else float("nan"),
            "scores": scores,
            "eigenfuncs": phi,
            "eigenvalues": ev[:kk],
            "mean": mu,
            "weights": w,
            "n": n,
            "method": "Karhunen-Loeve x_i(t) = mu(t) + sum_k s_ik phi_k(t) (Ramsay, Hooker and Graves 2009); not in the book",
        },
    )


def cheatsheet():
    return "fpca: Functional principal components analysis (FPCA)"


# compact alias per ledger/NAMING.md
functionalpca = functional_pca
