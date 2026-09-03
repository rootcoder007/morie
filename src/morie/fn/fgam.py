# morie.fn -- function file (rootcoder007/morie)
r"""The functional generalized additive model.

The functional LINEAR model forces the effect of the predictor curve to
be linear in :math:`X(t)` at every :math:`t`: doubling the curve doubles
its contribution. FGAM drops that and lets the contribution of the
observed level enter through an unknown *surface*,

.. math:: \mathbb{E}[Y \mid X] = \theta_0 + \int F\{X(t), t\}\,dt,

so the effect of being at level :math:`x` may differ in kind, not just in
scale, between early and late :math:`t`. Setting
:math:`F(x, t) = \beta(t)\,x` recovers the functional linear model
exactly, which is the comparison worth making and the one
``linear_deviation`` reports.

:math:`F` is expanded in a tensor product of cubic B-splines over
:math:`(x, t)` and fitted by penalised least squares with second-
difference penalties in each direction:

.. math:: \hat\theta = \arg\min \|y - \theta_0 - Z\theta\|^2
          + \lambda_x \theta' (D_x'D_x \otimes I)\theta
          + \lambda_t \theta' (I \otimes D_t'D_t)\theta,

with :math:`Z_{i,(j,l)} = \int b_j\{X_i(t)\}\,c_l(t)\,dt` evaluated on
the observation grid. The two penalties are separate on purpose: the
surface can be rough in :math:`x` and smooth in :math:`t`, or the other
way round, and a single smoothing parameter cannot express that.

**A surface is easy to overfit and easy to misread.** The effective
degrees of freedom :math:`\operatorname{tr}(H)` are returned so the fit
can be judged against its own flexibility rather than by eye.

References
----------
McLean, M. W., Hooker, G., Staicu, A.-M., Scheipl, F. and Ruppert, D.
(2014) "Functional generalized additive models", *Journal of
Computational and Graphical Statistics* **23**(1), 249-269,
doi:10.1080/10618600.2012.729985. The model, the tensor-product spline
representation of F and the penalised estimation.

Eilers, P. H. C. and Marx, B. D. (1996) "Flexible smoothing with
B-splines and penalties", *Statistical Science* **11**(2), 89-121,
doi:10.1214/ss/1038425655. The B-spline-plus-difference-penalty
construction used for both marginal bases.

Ramsay, J. O. and Silverman, B. W. (2005) *Functional Data Analysis*,
2nd ed., Springer, Ch. 15 (the functional linear model FGAM nests).
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["functional_gam"]

_EPS = 1e-12


def _grid_weights(n_t):
    if n_t < 2:
        return [1.0]
    h = 1.0 / (n_t - 1)
    w = [h] * n_t
    w[0] = 0.5 * h
    w[-1] = 0.5 * h
    return w


def _knots(lo, hi, n_basis, degree=3):
    """Clamped knot vector with equally spaced interior knots."""
    n_int = n_basis - degree - 1
    if n_int < 0:
        raise ValueError("fgam: a cubic basis needs at least %d functions"
                         % (degree + 1))
    span = hi - lo
    if span <= _EPS:
        span = 1.0
        hi = lo + 1.0
    inner = [lo + span * (j + 1.0) / (n_int + 1.0) for j in range(n_int)]
    return [lo] * (degree + 1) + inner + [hi] * (degree + 1)


def _bspline(x, kn, n_basis, degree=3):
    """Cox-de Boor evaluation of every basis function at x."""
    hi = kn[-1]
    if x >= hi:
        x = hi - _EPS
    if x <= kn[0]:
        x = kn[0] + _EPS
    B = [0.0] * (len(kn) - 1)
    for j in range(len(kn) - 1):
        if kn[j] <= x < kn[j + 1]:
            B[j] = 1.0
    for d in range(1, degree + 1):
        for j in range(len(kn) - d - 1):
            a = 0.0
            den1 = kn[j + d] - kn[j]
            if den1 > _EPS:
                a += (x - kn[j]) / den1 * B[j]
            den2 = kn[j + d + 1] - kn[j + 1]
            if den2 > _EPS:
                a += (kn[j + d + 1] - x) / den2 * B[j + 1]
            B[j] = a
    return B[:n_basis]


def _diff_penalty(n, order=2):
    """D'D for the order-th difference matrix on n coefficients."""
    D = []
    for i in range(n - order):
        row = [0.0] * n
        if order == 2:
            row[i], row[i + 1], row[i + 2] = 1.0, -2.0, 1.0
        else:
            row[i], row[i + 1] = -1.0, 1.0
        D.append(row)
    return [[sum(D[r][a] * D[r][b] for r in range(len(D)))
             for b in range(n)] for a in range(n)]


def functional_gam(X, Y, basis=None, n_x=6, n_t=6, lam_x=1.0, lam_t=1.0):
    r"""Fit E[Y|X] = theta0 + int F(X(t), t) dt.

    Parameters
    ----------
    X : (n, T) array-like
        Predictor curves on a common grid.
    Y : (n,) array-like
        Scalar response.
    basis : int, optional
        Convenience: sets both marginal basis sizes at once.
    n_x, n_t : int
        Cubic B-spline basis sizes in the level and time directions.
    lam_x, lam_t : float
        Second-difference penalties in each direction, separate because
        the surface may be rough in one and smooth in the other.
    """
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    y = [float(v) for v in k.vec(Y)]
    n = len(Xm)
    if n == 0:
        raise ValueError("fgam: no curves")
    if len(y) != n:
        raise ValueError("fgam: %d curves but %d responses" % (n, len(y)))
    T = len(Xm[0])
    if any(len(r) != T for r in Xm):
        raise ValueError("fgam: every curve must lie on the same grid")
    if basis is not None:
        n_x = n_t = int(basis)
    n_x, n_t = int(n_x), int(n_t)
    if n_x < 4 or n_t < 4:
        raise ValueError("fgam: each cubic marginal basis needs at least 4 "
                         "functions")
    w = _grid_weights(T)
    grid = [i / (T - 1.0) if T > 1 else 0.0 for i in range(T)]

    lo = min(min(r) for r in Xm)
    hi = max(max(r) for r in Xm)
    kx = _knots(lo, hi, n_x)
    kt = _knots(0.0, 1.0, n_t)
    Bt = [_bspline(t, kt, n_t) for t in grid]

    p = n_x * n_t
    Z = []
    for i in range(n):
        row = [0.0] * p
        for t in range(T):
            bx = _bspline(Xm[i][t], kx, n_x)
            wt = w[t]
            for a in range(n_x):
                if abs(bx[a]) <= _EPS:
                    continue
                base = a * n_t
                for b in range(n_t):
                    row[base + b] += wt * bx[a] * Bt[t][b]
        Z.append(row)

    ybar = sum(y) / n
    yc = [v - ybar for v in y]
    ZtZ = [[sum(Z[i][a] * Z[i][b] for i in range(n)) for b in range(p)]
           for a in range(p)]
    Zty = [sum(Z[i][a] * yc[i] for i in range(n)) for a in range(p)]

    Px = _diff_penalty(n_x)
    Pt = _diff_penalty(n_t)
    for a in range(n_x):
        for b in range(n_x):
            if abs(Px[a][b]) <= _EPS:
                continue
            for c in range(n_t):
                ZtZ[a * n_t + c][b * n_t + c] += lam_x * Px[a][b]
    for c in range(n_t):
        for d in range(n_t):
            if abs(Pt[c][d]) <= _EPS:
                continue
            for a in range(n_x):
                ZtZ[a * n_t + c][a * n_t + d] += lam_t * Pt[c][d]
    # Numerical floor, scaled to the matrix. A fixed absolute ridge is
    # scale-blind and leaves the system near-singular, so the coefficients
    # would not be identified to better than about 1e-5 -- reproducible
    # neither across implementations nor across runs.
    scale = sum(ZtZ[a][a] for a in range(p)) / p
    ridge = 1e-8 * scale if scale > _EPS else 1e-10
    for a in range(p):
        ZtZ[a][a] += ridge

    theta = k.cholsolve(ZtZ, Zty)
    fitted = [ybar + sum(Z[i][a] * theta[a] for a in range(p))
              for i in range(n)]
    resid = [y[i] - fitted[i] for i in range(n)]

    # effective degrees of freedom, tr(H) with H = Z (Z'Z + P)^-1 Z':
    # the diagonal of H is z_i' (Z'Z + P)^-1 z_i, so one solve per row.
    edf = 0.0
    for i in range(n):
        zi = [Z[i][a] for a in range(p)]
        sol = k.cholsolve(ZtZ, zi)
        edf += sum(zi[a] * sol[a] for a in range(p))

    sst = sum(v * v for v in yc)
    sse = sum(v * v for v in resid)
    r2 = 1.0 - sse / sst if sst > _EPS else 0.0

    # the surface on a modest grid, and how far it is from a linear one
    nx_out = 11
    xs = [lo + (hi - lo) * j / (nx_out - 1.0) for j in range(nx_out)]
    surface = []
    for xv in xs:
        bx = _bspline(xv, kx, n_x)
        surface.append([sum(bx[a] * Bt[t][b] * theta[a * n_t + b]
                            for a in range(n_x) for b in range(n_t))
                        for t in range(T)])
    lin = 0.0
    for t in range(T):
        col = [surface[j][t] for j in range(nx_out)]
        mx = sum(xs) / nx_out
        mc = sum(col) / nx_out
        den = sum((xs[j] - mx) ** 2 for j in range(nx_out))
        sl = (sum((xs[j] - mx) * (col[j] - mc) for j in range(nx_out)) / den
              if den > _EPS else 0.0)
        for j in range(nx_out):
            lin = max(lin, abs(col[j] - (mc + sl * (xs[j] - mx))))

    return RichResult(payload={
        "estimate": fitted,
        "fitted": fitted,
        "residuals": resid,
        "coefficients": theta,
        "intercept": ybar,
        "surface": surface,
        "surface_x": xs,
        "edf": edf,
        "r_squared": r2,
        "n_x": n_x,
        "n_t": n_t,
        "lam_x": float(lam_x),
        "lam_t": float(lam_t),
        "linear_deviation": lin,
        "n": n,
        "method": "functional generalized additive model, tensor-product "
                  "cubic B-splines with separate second-difference "
                  "penalties (McLean et al. 2014)",
        "note": "F(x, t) = beta(t) x recovers the functional linear model; "
                "linear_deviation is how far the fitted surface departs "
                "from that, so the extra flexibility is measured rather "
                "than assumed",
    })


def cheatsheet():
    return ("fgam: functional_gam(X, Y, n_x, n_t, lam_x, lam_t) -> "
            "E[Y|X] = theta0 + int F(X(t), t) dt by tensor-product "
            "penalised splines (McLean et al. 2014, JCGS 23(1), 249-269)")


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
functionalgam = functional_gam
