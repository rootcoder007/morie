# morie.fn -- function file (rootcoder007/morie)
"""Orthogonalized Gnanadesikan-Kettenring robust covariance."""

import math

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["orthogonalized_gk"]


def orthogonalized_gk(y, X=None):
    """Robust scatter by the OGK algorithm of Maronna & Zamar.

    The Gnanadesikan-Kettenring identity turns a covariance into a pair
    of scale estimates,

        cov(u, v) = (sigma(u + v)^2 - sigma(u - v)^2) / 4,

    so a robust scale gives a robust covariance -- entry by entry.  The
    catch is that a matrix built this way need not be positive
    semi-definite, which makes it useless as a scatter.  Maronna &
    Zamar's fix is to build it once, take its eigenvectors as a new
    basis, and rebuild the scatter DIAGONALLY in that basis:

      1. ``y_j = x_j / sigma_j`` with ``sigma_j`` a robust scale
         (MAD, so the estimate is affine equivariant under scaling);
      2. ``U_jk`` by the GK identity on the standardized columns,
         ``U_jj = 1``;
      3. ``U = E Lambda E'``, ``z = E' y``;
      4. ``Gamma = diag(sigma(z_l)^2)``, ``Sigma = A Gamma A'`` with
         ``A = diag(sigma) E``, which is positive semi-definite by
         construction;
      5. location ``mu = A m(z)`` with ``m`` the coordinatewise median.

    Determinism: MAD and medians only; no subsampling, no iteration.

    Parameters
    ----------
    y : array-like
        Either the full data matrix, shape (n, p), or -- when ``X`` is
        given -- the first column of it, shape (n,).
    X : array-like or None
        Remaining columns, shape (n, q).  When supplied, the data
        matrix is ``[y, X]`` and the robust regression coefficient
        ``Sigma_xx^{-1} Sigma_xy`` is returned as well.

    Returns
    -------
    RichResult
        ``sigma`` (the p by p scatter), ``location``, ``scales`` (the
        marginal MADs), ``estimate`` (``sigma[0][0]``), ``det``,
        ``beta`` (robust regression coefficients on ``X``, present only
        when ``X`` is given), ``n``, ``p``.

    References
    ----------
    Maronna, R. A. & Zamar, R. H. (2002).  Robust estimates of location
    and dispersion for high-dimensional datasets.  Technometrics,
    44(4), 307--317.  doi:10.1198/004017002188618509
    Gnanadesikan, R. & Kettenring, J. R. (1972).  Robust estimates,
    residuals, and outlier detection with multiresponse data.
    Biometrics, 28(1), 81--124.
    """
    if X is None:
        M = C.mat(y)
    else:
        yv = C.vec(y)
        Xm = C.mat(X)
        if len(Xm) != len(yv):
            raise ValueError("orthogonalized_gk: y and X have different lengths")
        M = [[yv[i]] + list(Xm[i]) for i in range(len(yv))]
    n = len(M)
    if n == 0:
        raise ValueError("orthogonalized_gk: data matrix is empty")
    p = len(M[0])
    if any(len(r) != p for r in M):
        raise ValueError("orthogonalized_gk: ragged data matrix")
    if n < 2:
        raise ValueError("orthogonalized_gk: need at least two observations")

    cols = [[M[i][j] for i in range(n)] for j in range(p)]
    sig = [core.mad(c) for c in cols]
    if any(s <= 0.0 for s in sig):
        raise ValueError("orthogonalized_gk: a column has zero robust scale")
    Y = [[M[i][j] / sig[j] for j in range(p)] for i in range(n)]

    U = [[0.0] * p for _ in range(p)]
    for j in range(p):
        U[j][j] = 1.0
        for k in range(j + 1, p):
            a = core.mad([Y[i][j] + Y[i][k] for i in range(n)])
            b = core.mad([Y[i][j] - Y[i][k] for i in range(n)])
            U[j][k] = U[k][j] = (a * a - b * b) / 4.0
    vals, vecs = core.jacobi(U)
    # jacobi returns columns of `vecs` as eigenvectors of U
    Z = [[sum(Y[i][j] * vecs[j][l] for j in range(p)) for l in range(p)]
         for i in range(n)]
    gam = [core.mad([Z[i][l] for i in range(n)]) ** 2 for l in range(p)]
    med = [core.median([Z[i][l] for i in range(n)]) for l in range(p)]
    A = [[sig[j] * vecs[j][l] for l in range(p)] for j in range(p)]
    S = [[sum(A[j][l] * gam[l] * A[k][l] for l in range(p)) for k in range(p)]
         for j in range(p)]
    mu = [sum(A[j][l] * med[l] for l in range(p)) for j in range(p)]

    out = {"sigma": S, "location": mu, "scales": sig,
           "estimate": S[0][0], "eigenvalues": vals,
           "n": n, "p": p,
           "method": "Orthogonalized Gnanadesikan-Kettenring scatter"}
    d = 1.0
    for l in range(p):
        d *= gam[l]
    for j in range(p):
        d *= sig[j] * sig[j]
    out["det"] = d
    if X is not None and p > 1:
        Sxx = [[S[j][k] for k in range(1, p)] for j in range(1, p)]
        Sxy = [S[j][0] for j in range(1, p)]
        out["beta"] = C.solvev(Sxx, Sxy)
    return RichResult(payload=out)


def cheatsheet():
    return "ogkcv: Orthogonalized Gnanadesikan-Kettenring robust scatter"


orthogonalizedgk = orthogonalized_gk
