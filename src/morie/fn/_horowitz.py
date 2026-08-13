# morie.fn -- function file (rootcoder007/morie)
"""Shared nonparametric/semiparametric machinery for the Horowitz shelf.

Spec: Horowitz, J. L., *Semiparametric and Nonparametric Methods in
Econometrics*, Springer.

The recurring theme of the book is that a nonparametric object
converges more slowly than a parametric one, yet a well-constructed
semiparametric FUNCTIONAL of it can still be root-n. Everything here
keeps the two apart: kernel objects report their bandwidth and the
rate it implies, while root-n functionals report a standard error.
"""

from . import _array_core as np

__all__ = [
    "kernel", "kernel_deriv", "silverman_bw", "kde", "kde_deriv",
    "nw_regression", "local_linear", "local_linear_quantile",
    "sieve_basis", "check_rate", "coord_min", "qirls",
]


def kernel(u, name="gaussian"):
    """Second-order kernels with unit variance conventions."""
    u = np.asarray(u, dtype=float)
    if name == "gaussian":
        return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    if name == "epanechnikov":
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    if name == "uniform":
        return np.where(np.abs(u) <= 1, 0.5, 0.0)
    raise ValueError(
        f"kernel must be 'gaussian', 'epanechnikov' or 'uniform', got {name!r}."
    )


def kernel_deriv(u, name="gaussian"):
    """First derivative K'(u), needed for density-derivative estimates."""
    u = np.asarray(u, dtype=float)
    if name == "gaussian":
        return -u * np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    if name == "epanechnikov":
        return np.where(np.abs(u) <= 1, -1.5 * u, 0.0)
    if name == "uniform":
        return np.zeros_like(u)  # flat: derivative is 0 a.e., undefined at +-1
    raise ValueError(f"unknown kernel {name!r}.")


def silverman_bw(x, factor=1.06):
    r"""Silverman's rule :math:`h = c\,\hat\sigma\,n^{-1/5}`.

    Silverman, B. W. (1986) *Density Estimation for Statistics and Data
    Analysis*, Monographs on Statistics and Applied Probability 26,
    Chapman & Hall, London, eq. (3.28) -- in the library.

    The exponent -1/5 is the MISE-optimal rate for a second-order
    kernel and a twice-differentiable density; it is NOT optimal for
    derivative estimation, which needs -1/7 (see
    :func:`kde_deriv`).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 observations.")
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    spread = min(float(x.std(ddof=1)), iqr / 1.349) if iqr > 0 else float(x.std(ddof=1))
    if spread <= 0:
        raise ValueError("x has zero spread; no bandwidth is defined.")
    return float(factor * spread * n ** (-0.2))


def kde(x, grid=None, h=None, name="gaussian"):
    r""":math:`\hat f(x) = \frac{1}{nh}\sum_i K((x - X_i)/h)`."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 observations.")
    h = silverman_bw(x) if h is None else float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    g = np.linspace(x.min() - 3 * h, x.max() + 3 * h, 512) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    dens = kernel((g[:, None] - x[None, :]) / h, name).sum(axis=1) / (n * h)
    return g, dens, h


def kde_deriv(x, grid=None, h=None, name="gaussian"):
    r""":math:`\hat f'(x) = -\frac{1}{nh^2}\sum_i K'((x - X_i)/h)`.

    Uses a wider bandwidth than the density itself: the optimal rate
    for the rth derivative is :math:`n^{-1/(2r+5)}`, so
    differentiating a density-optimal bandwidth undersmooths and the
    derivative estimate is far noisier than it needs to be.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 observations.")
    h = float(silverman_bw(x) * n ** (0.2 - 1.0 / 7.0)) if h is None else float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    g = np.linspace(x.min(), x.max(), 512) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    d = kernel_deriv((g[:, None] - x[None, :]) / h, name).sum(axis=1) / (n * h**2)
    return g, d, h


def nw_regression(x, y, grid=None, h=None, name="gaussian"):
    r"""Nadaraya-Watson:
    :math:`\hat m(x) = \sum_i K_h(x-X_i)Y_i / \sum_i K_h(x-X_i)`.

    A local CONSTANT fit, so it carries an O(h) boundary bias that
    local linear regression removes -- which is why the book prefers
    the latter near the support edges.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    h = silverman_bw(x) if h is None else float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    g = np.linspace(x.min(), x.max(), 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    W = kernel((g[:, None] - x[None, :]) / h, name)
    den = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(den > 0, (W @ y) / np.maximum(den, 1e-300), np.nan)
    return g, m, h


def local_linear(x, y, grid=None, h=None, name="gaussian"):
    r"""Local linear regression: minimise
    :math:`\sum_i K_h(x-X_i)(Y_i - a - b(X_i-x))^2`; return
    :math:`\hat m(x) = \hat a` and the slope :math:`\hat b`.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    h = silverman_bw(x) if h is None else float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    g = np.linspace(x.min(), x.max(), 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    m = np.empty(g.size)
    b = np.empty(g.size)
    for i, pt in enumerate(g):
        u = (x - pt) / h
        w = kernel(u, name)
        if w.sum() <= 0:
            m[i] = b[i] = np.nan
            continue
        X = np.column_stack([np.ones(x.size), x - pt])
        WX = X * w[:, None]
        try:
            coef = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(X.T @ WX, WX.T @ y, rcond=None)[0]
        m[i], b[i] = coef[0], coef[1]
    return g, m, b, h


def local_linear_quantile(x, y, tau=0.5, grid=None, h=None, name="gaussian",
                          n_iter=60):
    r"""Local linear quantile regression: minimise
    :math:`\sum_i K_h(x-X_i)\rho_\tau(Y_i - a - b(X_i-x))` by
    iteratively reweighted least squares on the check loss.
    """
    if not 0 < tau < 1:
        raise ValueError(f"tau must lie in (0, 1), got {tau}.")
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    h = silverman_bw(x) if h is None else float(h)
    g = np.linspace(x.min(), x.max(), 100) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    q = np.empty(g.size)
    for i, pt in enumerate(g):
        w0 = kernel((x - pt) / h, name)
        X = np.column_stack([np.ones(x.size), x - pt])
        coef = np.zeros(2)
        for _ in range(int(n_iter)):
            r = y - X @ coef
            # check-loss IRLS weights: tau above the fit, 1-tau below
            wq = np.where(r >= 0, tau, 1 - tau) / np.maximum(np.abs(r), 1e-6)
            W = w0 * wq
            try:
                new = np.linalg.solve(X.T @ (X * W[:, None]), (X * W[:, None]).T @ y)
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(new - coef)) < 1e-8:
                coef = new
                break
            coef = new
        q[i] = coef[0]
    return g, q, h


def sieve_basis(x, K=5, kind="poly"):
    """Sieve (series) basis of dimension K."""
    x = np.asarray(x, dtype=float).ravel()
    K = int(K)
    if K < 1:
        raise ValueError(f"K must be at least 1, got {K}.")
    lo, hi = float(x.min()), float(x.max())
    z = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    if kind == "poly":
        return np.column_stack([z**k for k in range(K)])
    if kind == "fourier":
        cols = [np.ones_like(z)]
        for k in range(1, K):
            cols.append(np.cos(np.pi * k * z) if k % 2 else np.sin(np.pi * k * z))
        return np.column_stack(cols[:K])
    raise ValueError("kind must be 'poly' or 'fourier'.")


def check_rate(errors, n_grid, expected_exponent):
    r"""Fit :math:`\log \text{error} = c + \gamma \log n` and compare
    the observed exponent gamma against the theoretical one.

    Rate theorems have no single-sample value; this returns the
    measured exponent so the claim is checkable rather than asserted.
    """
    e = np.asarray(errors, dtype=float).ravel()
    n = np.asarray(n_grid, dtype=float).ravel()
    if e.size != n.size or e.size < 3:
        raise ValueError("need at least 3 matched (error, n) pairs.")
    if np.any(e <= 0) or np.any(n <= 0):
        raise ValueError("errors and sample sizes must be positive.")
    slope, intercept = np.polyfit(np.log(n), np.log(e), 1)
    return {
        "observed_exponent": float(slope),
        "expected_exponent": float(expected_exponent),
        "intercept": float(intercept),
        "consistent": bool(abs(slope - expected_exponent) < 0.15),
    }


GRID_SCAN_HALF_WIDTH = 10.0
GRID_SCAN_POINTS = 2001


def optimize_scale_normalized(objective, d, n_restarts=8, seed=0, x0=None):
    r"""Minimise ``objective(b)`` over :math:`b` with :math:`b_1 = 1`.

    Every maximum-score variant in Chapter 4 is identified only up to
    scale, so the book normalises :math:`|b_1| = 1` and optimises over
    the remaining ``d - 1`` coordinates.

    These objectives are STEP FUNCTIONS of ``b``, which rules out any
    gradient method and makes simplex methods start-dependent: they
    stall on the first flat region they land in. For the ``d = 2``
    case -- one free coordinate, and the usual one -- this uses an
    exhaustive grid scan instead, which is both the right method for
    a piecewise-constant objective and exactly reproducible, so the
    Python and R implementations agree to the last digit rather than
    to whatever their respective simplex routines happen to do. The
    grid spans ``+/-10`` at a resolution of ``0.01``.

    For ``d > 2`` an exhaustive scan is not affordable and multi-start
    Nelder-Mead is used; there the restarts are not defensive padding
    but the only thing making the answer reproducible.

    ``objective`` is always MINIMISED, so callers maximising a score
    pass its negation.
    """
    from ._sci_core import optimize as _opt

    d = int(d)
    if d < 2:
        raise ValueError(f"need at least 2 coefficients, got {d}.")
    if d == 2:
        grid = np.linspace(-GRID_SCAN_HALF_WIDTH, GRID_SCAN_HALF_WIDTH,
                           GRID_SCAN_POINTS)
        vals = np.array([objective(np.array([1.0, g])) for g in grid])
        k = int(np.argmin(vals))
        return np.array([1.0, float(grid[k])]), float(vals[k])
    rng = np.random.default_rng(seed)
    starts = [np.zeros(d - 1) if x0 is None else
              np.atleast_1d(np.asarray(x0, dtype=float)).ravel()[-(d - 1):]]
    starts += [rng.standard_normal(d - 1) for _ in range(int(n_restarts))]
    best, best_val = None, np.inf
    for st in starts:
        r = _opt.minimize(lambda z: objective(np.r_[1.0, z]), st,
                          method="Nelder-Mead",
                          options={"maxiter": 3000, "fatol": 1e-9})
        if r.fun < best_val:
            best_val, best = float(r.fun), r.x
    return np.r_[1.0, best], best_val


def cheatsheet():
    return "_horowitz: kernels, KDE, NW, local linear/quantile, sieves, rate checks"


def coord_min(fun, x0, niter=12, delta=1.0, shrink=0.5, steps=3):
    """Deterministic coordinate search for a smooth low-dimensional
    objective.

    A FIXED schedule: `niter` sweeps, each trying offsets
    (-steps ... +steps) * delta on every coordinate and keeping the
    best, with delta multiplied by `shrink` after each sweep.  There
    is NO tolerance-based early exit and no random restart, so the
    same inputs give the same answer in every language this is
    mirrored into -- which is the whole point.
    """
    x = [float(v) for v in x0]
    best = float(fun(x))
    d = float(delta)
    for _ in range(int(niter)):
        for j in range(len(x)):
            base = x[j]
            for k in range(-int(steps), int(steps) + 1):
                if k == 0:
                    continue
                x[j] = base + k * d
                val = float(fun(x))
                if val < best:
                    best = val
                    base = x[j]
            x[j] = base
        d = d * float(shrink)
    return x, best


def qirls(X, y, w, tau, niter=40, eps=1e-3):
    """Fixed-iteration IRLS for the check loss rho_tau.

    Fixed iteration count and NO tolerance-based early exit, so the
    Python and R arms take exactly the same path.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    p = X.shape[1]
    beta = np.zeros(p)
    for _ in range(int(niter)):
        r = y - X @ beta
        # The check-loss weight is DISCONTINUOUS at r = 0.  Where the
        # residual is already inside the eps floor the sign test is
        # decided by machine noise, and 40 iterations amplify that into
        # a visible cross-language difference, so a residual within eps
        # of zero is treated as a tie and given the average weight.
        num = np.where(np.abs(r) < eps, 0.5,
                       np.where(r > 0, tau, 1.0 - tau))
        wk = w * num / np.maximum(np.abs(r), eps)
        A = X.T @ (X * wk[:, None]) + 1e-10 * np.eye(p)
        b = X.T @ (wk * y)
        # Jacobi equilibration.  A series design of powers is badly
        # conditioned, and solving it raw lets two LAPACK paths differ
        # in the last bits -- which 40 IRLS steps then amplify into a
        # visible cross-language gap.  Scaling by the square roots of
        # the diagonal is exact arithmetic and fixes it.
        dg = np.sqrt(np.maximum(np.diag(A), 1e-300))
        beta = np.linalg.solve(A / dg[:, None] / dg[None, :], b / dg) / dg
    return beta
