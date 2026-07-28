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

import numpy as np

__all__ = [
    "kernel", "kernel_deriv", "silverman_bw", "kde", "kde_deriv",
    "nw_regression", "local_linear", "local_linear_quantile",
    "sieve_basis", "check_rate",
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


def cheatsheet():
    return "_horowitz: kernels, KDE, NW, local linear/quantile, sieves, rate checks"
