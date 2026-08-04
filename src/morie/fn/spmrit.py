# morie.fn -- function file (rootcoder007/morie)
"""Moran's I on OLS residuals, eq (1.16)."""

from math import fsum, sqrt

from ._richresult import RichResult
from ._spx import eye, mat, matmul, solve, sqmat, trace, transpose, twosidep, vec

__all__ = [
    "schabenberger_moran_i_residuals",
    "moranres",
]


def schabenberger_moran_i_residuals(residuals, w, x=None):
    """Moran's I for OLS residuals, Schabenberger & Gotway eq (1.16).

    Sec. 1.3.2 warns that Moran's I on raw data confounds autocorrelation
    with a non-constant mean -- Example 1.7 shows I = 0.2597 with
    p = 0.00011 on data that have NO autocorrelation at all, only a
    trending mean. The remedy the book gives is to fit the mean model
    ``Z(s) = X(s)beta + e``, take OLS residuals, and use

        Ires = n e'W e / (w.. e'e)                              eq (1.16)

    with ehat = M e, ``M = I - X(X'X)^-1 X'``. The book prints the mean,

        Eg[Ires] = n tr[MW] / {(n-k) w..},

    and that expression is reproduced here exactly (returned as
    ``expectation``); it is NOT the -1/(n-1) of the raw statistic.

    The generated stub attributed this to "Schabenberger Ch 6" and computed
    a Spearman correlation. Chapter 6 is spatial regression; eq (1.16) is
    in CHAPTER 1, Sec. 1.3.2.

    The variance is not printed in the book and is derived here. Writing
    A = (W + W')/2 and B = MAM (so BM = B and MB = B), Ires = (n/w..) T
    with T = eps'B eps / eps'M eps for eps ~ G(0, sigma^2 I). For two
    quadratic forms in the same Gaussian projection the exact moments are

        E[T]   = tr(B) / (n-k)
        E[T^2] = {2 tr(B^2) + tr(B)^2} / {(n-k)(n-k+2)},

    so Var[Ires] = (n/w..)^2 (E[T^2] - E[T]^2). tr(B) = tr(MW), which is
    why ``expectation`` agrees with the book's formula term for term.

    With `x` omitted the mean model is the intercept alone (k = 1), i.e.
    M = I - 11'/n; that is the right default only if the caller already
    removed a mean, and it is reported as ``k``.

    Parameters
    ----------
    residuals : (n,) array-like
        OLS residuals, or the raw attribute when `x` is supplied.
    w : (n, n) array-like
        Spatial weights with a zero diagonal.
    x : (n, k) array-like, optional
        Mean-model design matrix, intercept column included.

    Returns
    -------
    RichResult
        ``i``, ``expectation``, ``variance``, ``z``, ``p_value``, ``s0``,
        ``tr_mw``, ``k``, ``n``, ``method``.
    """
    e = vec(residuals, "residuals")
    n = len(e)
    if n < 4:
        raise ValueError("at least 4 sites are needed")
    ww = sqmat(w, n, "w")
    for i in range(n):
        if ww[i][i] != 0.0:
            raise ValueError("`w` must have a zero diagonal")
    s0 = fsum([fsum(row) for row in ww])
    if s0 <= 0:
        raise ValueError("total weight w.. must be positive")

    if x is None:
        k = 1
        proj = eye(n)
        for i in range(n):
            for j in range(n):
                proj[i][j] = proj[i][j] - 1.0 / n
    else:
        xx = mat(x, "x")
        if len(xx) != n:
            raise ValueError("`x` has %d rows but `residuals` has %d values"
                             % (len(xx), n))
        k = len(xx[0])
        if n - k < 3:
            raise ValueError("need n - k >= 3 residual degrees of freedom")
        xt = transpose(xx)
        g = matmul(xt, xx)
        inv = [solve(g, [1.0 if r == c else 0.0 for r in range(k)])
               for c in range(k)]
        inv = transpose(inv)
        proj = eye(n)
        h = matmul(matmul(xx, inv), xt)
        for i in range(n):
            for j in range(n):
                proj[i][j] = proj[i][j] - h[i][j]

    ee = fsum([t * t for t in e])
    if ee <= 0:
        raise ValueError("the residuals are all zero; Ires is undefined")
    ewe = fsum([ww[i][j] * e[i] * e[j] for i in range(n) for j in range(n)])
    ires = n * ewe / (s0 * ee)

    a = [[0.5 * (ww[i][j] + ww[j][i]) for j in range(n)] for i in range(n)]
    b = matmul(matmul(proj, a), proj)
    trb = trace(b)
    trb2 = trace(matmul(b, b))
    df = float(n - k)
    scale = n / s0
    et = trb / df
    et2 = (2.0 * trb2 + trb * trb) / (df * (df + 2.0))
    ex = scale * et
    var = scale * scale * (et2 - et * et)
    if var <= 0:
        raise ValueError("the null variance of Ires is not positive")
    zz = (ires - ex) / sqrt(var)

    return RichResult(payload={
        "i": ires,
        "expectation": ex,
        "variance": var,
        "z": zz,
        "p_value": twosidep(zz),
        "s0": s0,
        "tr_mw": trb,
        "k": k,
        "n": n,
        "not_minus_one_over_n_minus_one": True,
        "method": ("Moran's I on OLS residuals, Schabenberger & Gotway "
                   "(2005) eq (1.16) with Eg[Ires] as printed in "
                   "Sec. 1.3.2; the variance is derived"),
    })


def cheatsheet():
    return "spmrit: Moran's I on OLS residuals, eq (1.16)"


# compact alias per ledger/NAMING.md
moranres = schabenberger_moran_i_residuals
