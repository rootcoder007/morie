# morie.fn -- function file (rootcoder007/morie)
"""Robust factor analysis on a minimum covariance determinant scatter.

Pison, G., Rousseeuw, P. J., Filzmoser, P. and Croux, C. (2003),
"Robust factor analysis", *Journal of Multivariate Analysis* 84(1),
145-172.  The proposal of that paper is the one named in the stub
docstring: run the classical factor-analytic machinery, but on a
high-breakdown scatter matrix rather than on the sample covariance,
the MCD being the estimator used.  The factor model itself is
unchanged,

    Sigma = Lambda Lambda' + Psi,

with Lambda the p-by-k loading matrix and Psi the diagonal matrix of
uniquenesses; substituting the robust scatter for Sigma is what makes
the analysis resistant.

Extraction here is principal factor analysis on the robust CORRELATION
matrix implied by that scatter: the loadings are the first k
eigenvectors scaled by the square roots of their eigenvalues, so that
Lambda Lambda' is the rank-k eigen-approximation of the correlation
matrix, and the uniquenesses are the leftover diagonal.  The
eigenproblem is the cyclic Jacobi routine shared with the rest of this
package, whose eigenvector signs are fixed so that the two language
arms agree; without that fix the loadings would differ by a sign
column-wise and parity would fail on a correct implementation.

Two exact consequences serve as anchors, neither of which runs through
the extraction:

  * at k = p the rank-k approximation is the whole spectral
    decomposition, so Lambda Lambda' reproduces the correlation matrix
    exactly and every uniqueness is zero;
  * for uncorrelated variables the robust correlation matrix is the
    identity, whose eigenvalues are all 1, so the communalities are
    k/p per variable and the reproduced off-diagonals are zero.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _rousscore as R
from . import _s03core as k
from .mcdv import mcd

from ._richresult import RichResult

__all__ = ["robust_factor_analysis"]


def robust_factor_analysis(X, k_factors=1, h=None, max_subsets=200000):
    """Principal factor analysis on the MCD correlation matrix.

    Parameters
    ----------
    X : array-like
        n-by-p data matrix.
    k_factors : int
        Number of factors, 1 <= k <= p.
    h : int, optional
        MCD subset size; defaults to [(n + p + 1) / 2].
    max_subsets : int
        Passed to the MCD enumeration.

    Returns
    -------
    estimate : the first communality
    loadings : p-by-k matrix Lambda
    uniquenesses : the diagonal of Psi
    communalities : row sums of Lambda^2
    correlation : the robust correlation matrix
    reproduced : Lambda Lambda' + Psi
    eigenvalues : all p eigenvalues, descending
    """
    Xm = k.mat(X)
    n = k.nrow(Xm)
    if n == 0:
        raise ValueError("robust_factor_analysis: X is empty")
    p = k.ncol(Xm)
    if p == 0:
        raise ValueError("robust_factor_analysis: X has no columns")
    kf = int(k_factors)
    if kf < 1 or kf > p:
        raise ValueError("robust_factor_analysis: need 1 <= k_factors <= p")
    m = mcd(Xm, h, None, max_subsets)
    S = m["cov_raw"]
    sd = []
    for a in range(p):
        if S[a][a] <= 0.0:
            raise ValueError("robust_factor_analysis: a variable has zero robust variance")
        sd.append(math.sqrt(S[a][a]))
    Cm = [[S[a][b] / (sd[a] * sd[b]) for b in range(p)] for a in range(p)]
    vals, vecs = k.jacobi(Cm)
    # k.jacobi returns eigenvalues ascending; take them descending
    order = list(range(p - 1, -1, -1))
    evals = [vals[i] for i in order]
    L = [[0.0] * kf for _ in range(p)]
    for c in range(kf):
        lam = evals[c]
        s = math.sqrt(lam) if lam > 0.0 else 0.0
        src = order[c]
        for a in range(p):
            L[a][c] = vecs[a][src] * s
    comm = []
    for a in range(p):
        t = 0.0
        for c in range(kf):
            t += L[a][c] * L[a][c]
        comm.append(t)
    uniq = [Cm[a][a] - comm[a] for a in range(p)]
    rep = [[0.0] * p for _ in range(p)]
    for a in range(p):
        for b in range(p):
            t = 0.0
            for c in range(kf):
                t += L[a][c] * L[b][c]
            rep[a][b] = t + (uniq[a] if a == b else 0.0)
    return RichResult(
        title="Robust factor analysis",
        summary_lines=[("n", n), ("p", p), ("factors", kf), ("first communality", comm[0])],
        payload={
            "estimate": comm[0],
            "loadings": L,
            "uniquenesses": uniq,
            "communalities": comm,
            "correlation": Cm,
            "reproduced": rep,
            "eigenvalues": evals,
            "center": m["center"],
            "k_factors": kf,
            "n": n,
            "p": p,
            "method": "Pison-Rousseeuw-Filzmoser-Croux (2003) robust factor analysis: principal factors on the MCD correlation matrix",
        },
    )


def cheatsheet():
    return "rfcomp: robust factor analysis on the MCD scatter"
