# morie.fn -- function file (rootcoder007/morie)
"""Thin-plate spline surface plus linear covariates."""

from math import fsum, log

from ._richresult import RichResult
from ._spx import eucdist, mat, solve, vec

__all__ = [
    "spatial_gams",
    "spgam",
    "spatialgams",
]


def spatial_gams(y, x, coords, lam=0.0):
    """Semiparametric fit ``y = f(s) + X beta`` with f a thin-plate spline.

    NOT IN SCHABENBERGER & GOTWAY -- a fixed-string search for
    "generalized additive" returns nothing; the book's parametric
    counterpart is the trend surface of Sec. 5.3.1. The smoother is the
    thin-plate spline of Duchon, J. (1977), "Splines minimizing
    rotation-invariant semi-norms in Sobolev spaces", in *Constructive
    Theory of Functions of Several Variables*, Springer, pp. 85-100, as
    presented in Wood, S. N. (2006), *Generalized Additive Models: An
    Introduction with R*, Chapman & Hall/CRC, Ch. 4. Named from the
    general literature; NOT verified against a PDF in this corpus.

    In two dimensions the thin-plate basis is ``eta(r) = r^2 log r``
    (with eta(0) = 0), and the fit solves the saddle-point system

        [ K + n lam I   T ] [c]   [y]
        [ T'            0 ] [d] = [0]

    where K_ij = eta(||s_i - s_j||) and T = [1, s1, s2, X] carries the
    null space of the penalty -- the affine trend and the linear
    covariates, which the roughness penalty must NOT shrink. Dropping the
    T' c = 0 block leaves the system singular and is the usual failure.

    ``lam = 0`` interpolates exactly (residuals are zero to rounding);
    positive lam smooths. lam is a FIXED argument, not chosen by GCV: a
    GCV search needs the trace of the hat matrix, which for this
    saddle-point system costs n further solves, and no effective-degrees-
    of-freedom number is reported rather than reporting a guessed one. The
    roughness penalty c'Kc at the chosen lam IS returned, so a caller can
    scan lam and watch fit trade against roughness.

    Parameters
    ----------
    y : (n,) array-like
        Response.
    x : (n, k) array-like or None
        Linear covariates WITHOUT an intercept column (one is added).
    coords : (n, 2) array-like
        Site coordinates.
    lam : float
        Smoothing parameter, non-negative.

    Returns
    -------
    RichResult
        ``fitted``, ``residuals``, ``coef``, ``spline_weights``, ``rss``,
        ``penalty``, ``n``, ``method``.
    """
    yv = vec(y, "y")
    n = len(yv)
    cc = mat(coords, "coords")
    if len(cc) != n:
        raise ValueError("`coords` has %d rows but `y` has %d values"
                         % (len(cc), n))
    if len(cc[0]) < 2:
        raise ValueError("`coords` must have two columns for a 2-D "
                         "thin-plate spline")
    lam = float(lam)
    if lam < 0:
        raise ValueError("`lam` must be non-negative")
    if x is None:
        xm = [[] for _ in range(n)]
    else:
        xm = mat(x, "x")
        if len(xm) != n:
            raise ValueError("`x` has %d rows but `y` has %d values"
                             % (len(xm), n))
    t = [[1.0, cc[i][0], cc[i][1]] + list(xm[i]) for i in range(n)]
    m = len(t[0])
    if n <= m:
        raise ValueError("need more sites than null-space columns (%d)" % m)

    k = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                r = eucdist(cc[i][:2], cc[j][:2])
                k[i][j] = r * r * log(r) if r > 0 else 0.0

    size = n + m
    a = [[0.0] * size for _ in range(size)]
    for i in range(n):
        for j in range(n):
            a[i][j] = k[i][j]
        a[i][i] = a[i][i] + n * lam
        for j in range(m):
            a[i][n + j] = t[i][j]
            a[n + j][i] = t[i][j]
    rhs = list(yv) + [0.0] * m
    sol = solve(a, rhs)
    c = sol[:n]
    d = sol[n:]

    fitted = [fsum([k[i][j] * c[j] for j in range(n)])
              + fsum([t[i][j] * d[j] for j in range(m)]) for i in range(n)]
    resid = [yv[i] - fitted[i] for i in range(n)]
    rss = fsum([r_ * r_ for r_ in resid])
    penalty = fsum([c[i] * fsum([k[i][j] * c[j] for j in range(n)])
                    for i in range(n)])

    return RichResult(payload={
        "fitted": fitted,
        "residuals": resid,
        "coef": d,
        "spline_weights": c,
        "rss": rss,
        "penalty": penalty,
        "lam": lam,
        "null_space_is_unpenalised": True,
        "n": n,
        "method": ("Thin-plate spline surface plus linear covariates "
                   "(Duchon 1977; Wood 2006, Ch. 4); NOT in Schabenberger "
                   "& Gotway, whose parametric analogue is Sec. 5.3.1"),
    })


def cheatsheet():
    return "spgams: thin-plate spline surface plus linear covariates"


# compact alias per ledger/NAMING.md
spgam = spatial_gams
spatialgams = spatial_gams
