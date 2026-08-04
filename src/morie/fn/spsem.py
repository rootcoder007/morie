# morie.fn -- function file (rootcoder007/morie)
"""SAR error model by maximum likelihood, eqs (6.35)-(6.37)."""

from math import fsum, log, pi, sqrt

from ._richresult import RichResult
from ._spx import (dot, eye, logabsdet, lstsq, mat, matvec, sqmat, vec)

__all__ = [
    "schabenberger_spatial_error_model",
    "sperrmod",
]


def _neg2ll(y, x, w, rho):
    n = len(y)
    a = eye(n)
    for i in range(n):
        for j in range(n):
            a[i][j] = a[i][j] - rho * w[i][j]
    sign, lad = logabsdet(a)
    if sign <= 0 or lad == float("-inf"):
        return float("inf"), None, float("nan")
    ys = matvec(a, y)
    xs = [matvec(a, [x[i][k] for i in range(n)]) for k in range(len(x[0]))]
    xs = [[xs[k][i] for k in range(len(xs))] for i in range(n)]
    beta = lstsq(xs, ys)
    r = [ys[i] - dot(xs[i], beta) for i in range(n)]
    s2 = dot(r, r) / n
    if s2 <= 0:
        return float("inf"), None, float("nan")
    return n * log(2.0 * pi * s2) + n - 2.0 * lad, beta, s2


def schabenberger_spatial_error_model(x, y, w, n_grid=201, refine=60):
    """Simultaneous autoregressive ERROR model by ML, Sec. 6.2.2.1.

    Schabenberger & Gotway (2005) write the one-parameter SAR model as

        Z(s) = X(s) beta + e(s),   e(s) = rho W e(s) + upsilon   eq (6.36)
        Z(s) = X(s) beta + (I - rho W)^-1 upsilon                eq (6.37)

    which induces, from eq (6.35) with B = rho W and Sigma_upsilon =
    sigma^2 I,

        Sigma_SAR = sigma^2 (I - rho W)^-1 (I - rho W')^-1.

    The autocorrelation sits entirely in the errors, so beta keeps its
    ordinary regression interpretation. This is NOT the spatially lagged
    model of eq (6.38); the two share the name "SAR" in different
    literatures and give different beta.

    Whitening both sides by A = I - rho W turns the likelihood into an
    ordinary least-squares problem at each fixed rho, so beta and sigma^2
    profile out and only a ONE-dimensional search remains:

        -2 logL(rho) = n log(2 pi sigma^2(rho)) + n - 2 log|I - rho W|.

    That is why a grid scan plus golden-section refinement is used rather
    than a general optimiser: the profile is one-dimensional and the
    Jacobian term -2 log|A| is what makes naive OLS-in-rho wrong.

    The admissible interval for rho is the non-singularity condition of
    Sec. 6.2.2.1, p. 336: ``1/theta_min < rho < 1/theta_max`` with
    theta the eigenvalues of W. For symmetric W that is contained in
    ``|rho| < 1/rho(W)``, computed here by power iteration and shrunk by
    ``pad`` so the endpoint, where log|A| diverges, is never evaluated.

    The generated stub ran a Spearman correlation on `x` against `y` and
    ignored `w` entirely.

    Parameters
    ----------
    x : (n, k) array-like
        Design matrix; supply the intercept column yourself.
    y : (n,) array-like
        Response.
    w : (n, n) array-like
        Symmetric spatial weights, zero diagonal.
    n_grid : int
        Grid points in the initial scan.
    refine : int
        Golden-section iterations after the scan.

    Returns
    -------
    RichResult
        ``rho``, ``beta``, ``sigma2``, ``neg2loglik``, ``rho_bounds``,
        ``ols_beta``, ``k``, ``n``, ``method``.
    """
    yy = vec(y, "y")
    n = len(yy)
    xx = mat(x, "x")
    if len(xx) != n:
        raise ValueError("`x` has %d rows but `y` has %d values"
                         % (len(xx), n))
    k = len(xx[0])
    if n <= k + 1:
        raise ValueError("need n > k + 1 observations")
    ww = sqmat(w, n, "w")
    for i in range(n):
        if ww[i][i] != 0.0:
            raise ValueError("`w` must have a zero diagonal")
        for j in range(i + 1, n):
            if abs(ww[i][j] - ww[j][i]) > 1e-12:
                raise ValueError("`w` must be symmetric for the eigenvalue "
                                 "bound of Sec. 6.2.2.1 to reduce to the "
                                 "spectral radius")
    n_grid = int(n_grid)
    if n_grid < 5:
        raise ValueError("`n_grid` must be at least 5")

    v = [float((i % 7) + 1) for i in range(n)]
    s = sqrt(dot(v, v))
    v = [t / s for t in v]
    for _ in range(400):
        u = matvec(ww, v)
        s = sqrt(dot(u, u))
        if s < 1e-300:
            raise ValueError("`w` is numerically zero; no neighbours")
        v = [t / s for t in u]
    srad = abs(dot(v, matvec(ww, v)))
    if srad <= 0:
        raise ValueError("`w` has spectral radius 0; rho is unidentified")
    hi = (1.0 / srad) * (1.0 - 1e-6)
    lo = -hi

    best = None
    for g in range(n_grid):
        rho = lo + (hi - lo) * g / (n_grid - 1.0)
        val, beta, s2 = _neg2ll(yy, xx, ww, rho)
        if best is None or val < best[0]:
            best = (val, rho, beta, s2)
    step = (hi - lo) / (n_grid - 1.0)
    a = max(lo, best[1] - step)
    b = min(hi, best[1] + step)
    inv = 0.6180339887498949
    c = b - inv * (b - a)
    d = a + inv * (b - a)
    fc = _neg2ll(yy, xx, ww, c)[0]
    fd = _neg2ll(yy, xx, ww, d)[0]
    for _ in range(int(refine)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv * (b - a)
            fc = _neg2ll(yy, xx, ww, c)[0]
        else:
            a, c, fc = c, d, fd
            d = a + inv * (b - a)
            fd = _neg2ll(yy, xx, ww, d)[0]
    rho = 0.5 * (a + b)
    val, beta, s2 = _neg2ll(yy, xx, ww, rho)
    if beta is None:
        raise ValueError("the likelihood is undefined at the optimum; "
                         "check that `w` admits a non-singular I - rho W")
    ols = lstsq(xx, yy)

    return RichResult(payload={
        "rho": rho,
        "beta": beta,
        "sigma2": s2,
        "neg2loglik": val,
        "rho_bounds": [lo, hi],
        "ols_beta": ols,
        "spectral_radius": srad,
        "is_error_model_not_lag_model": True,
        "k": k,
        "n": n,
        "method": ("SAR error model by ML, Schabenberger & Gotway (2005) "
                   "eqs (6.35)-(6.37), Sec. 6.2.2.1; concentrated "
                   "likelihood, grid scan + golden section"),
    })


def cheatsheet():
    return "spsem: SAR error model ML, eqs (6.35)-(6.37)"


# compact alias per ledger/NAMING.md
sperrmod = schabenberger_spatial_error_model
