# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
r"""Additive model via marginal integration.

Estimates an additive model :math:`m(x) = c + \\sum_j m_j(x_j)` using
marginal integration (Linton & Nielsen 1995): each component
:math:`m_j` is estimated by averaging the full (pilot) regression
surface over all other dimensions.

References
----------
Linton, O. & Nielsen, J. P. (1995). A kernel method of estimating
    structured nonparametric regression based on marginal integration.
    *Biometrika*, 82(1), 93--100.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 2.
"""

from __future__ import annotations

import math
from typing import Any


def _interp(x, xp, fp):
    """Piecewise-linear interpolation on an increasing grid, constant beyond
    the ends (np.interp / R approx(rule = 2))."""
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    lo, hi = 0, len(xp) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] + t * (fp[hi] - fp[lo])


def admod(
    Y,
    X,
    *,
    bandwidth: float | None = None,
    grid_size: int = 50,
) -> dict[str, Any]:
    r"""Additive model via marginal integration.

    The pilot is the full :math:`p`-dimensional Nadaraya-Watson estimate
    :math:`\hat g(x) = \sum_i K_h(x - X_i) Y_i / \sum_i K_h(x - X_i)` with a
    product Gaussian kernel. Component :math:`j` averages it over the
    observed values of the *other* covariates (Linton and Nielsen 1995,
    eq. 2):

    .. math::

        \hat m_j(t) = \frac1n \sum_{k=1}^n
            \hat g\big(t, X_{k,-j}\big) - \hat\mu, \qquad
        \hat\mu = \bar Y,

    which identifies :math:`m_j` under :math:`E\,m_j(X_j) = 0`. (Smoothing
    ``Y`` on ``X_j`` alone -- what this function used to do -- estimates
    :math:`E(Y \mid X_j)`, not the additive component, whenever the
    covariates are dependent.) Fitted values interpolate each component
    linearly on its grid.

    Parameters
    ----------
    Y : array-like
        Outcome vector, shape ``(n,)``.
    X : array-like
        Covariate matrix, shape ``(n, p)``.
    bandwidth : float or None
        Kernel bandwidth (all directions); if *None*,
        ``1.06 mean_j(sd_j) n^{-1/(4+p)}``.
    grid_size : int
        Number of grid points per component.

    Returns
    -------
    dict[str, Any]
        ``intercept``, ``components`` (list of dicts with ``x_grid``
        and ``m_hat``), ``residuals``, ``bandwidth``, ``n``, ``p``,
        ``method``.

    References
    ----------
    Linton, O. and Nielsen, J. P. (1995). A kernel method of estimating
    structured nonparametric regression based on marginal integration.
    Biometrika 82(1), 93-100.

    Examples
    --------
    >>> X = [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [4.0, 3.0], [5.0, 2.0]]
    >>> Y = [1.0, 1.5, 3.2, 3.9, 6.1, 6.0]
    >>> r = admod(Y, X, bandwidth=1.0, grid_size=3)
    >>> [round(v, 10) for v in r["components"][0]["m_hat"]]
    [-2.1929007165, -0.0184122108, 2.0961820347]
    """
    y = [float(v) for v in (Y.tolist() if hasattr(Y, "tolist") else Y)]
    Xl = X.tolist() if hasattr(X, "tolist") else [list(r) if isinstance(r, (list, tuple)) else r for r in X]
    if Xl and not isinstance(Xl[0], (list, tuple)):
        Xl = [[v] for v in Xl]
    Xl = [[float(v) for v in r] for r in Xl]
    n, p = len(Xl), len(Xl[0])
    if len(y) != n:
        raise ValueError(f"Y length {len(y)} != X rows {n}.")
    if bandwidth is None:
        sds = []
        for j in range(p):
            col = [r[j] for r in Xl]
            m = sum(col) / n
            sds.append(math.sqrt(sum((v - m) ** 2 for v in col) / n))
        bandwidth = 1.06 * (sum(sds) / p) * n ** (-1.0 / (4 + p))
    h = float(bandwidth)
    if not h > 0:
        raise ValueError("bandwidth must be positive")
    mu = 0.0
    for v in y:
        mu += v
    mu /= n
    components = []
    for j in range(p):
        # kernel weight between averaging point k and datum i in the other directions
        W = [[0.0] * n for _ in range(n)]
        for k in range(n):
            for i in range(n):
                s = 0.0
                for d in range(p):
                    if d != j:
                        u = (Xl[k][d] - Xl[i][d]) / h
                        s += u * u
                W[k][i] = math.exp(-0.5 * s)
        col = [r[j] for r in Xl]
        lo, hi = min(col), max(col)
        grid = [lo + (hi - lo) * g / (grid_size - 1) for g in range(grid_size)] if grid_size > 1 else [lo]
        mh = []
        for t in grid:
            A = [math.exp(-0.5 * ((t - col[i]) / h) ** 2) for i in range(n)]
            AY = [A[i] * y[i] for i in range(n)]
            tot = 0.0
            for k in range(n):
                num = 0.0
                den = 0.0
                Wk = W[k]
                for i in range(n):
                    num += Wk[i] * AY[i]
                    den += Wk[i] * A[i]
                tot += num / den if den > 1e-300 else mu
            mh.append(tot / n - mu)
        components.append({"x_grid": grid, "m_hat": mh})
    residuals = []
    for i in range(n):
        f = mu
        for j in range(p):
            f += _interp(Xl[i][j], components[j]["x_grid"], components[j]["m_hat"])
        residuals.append(y[i] - f)
    return {
        "intercept": mu,
        "components": components,
        "residuals": residuals,
        "bandwidth": h,
        "n": n,
        "p": p,
        "method": "AdditiveModel_MarginalIntegration",
    }


admod_fn = admod


def cheatsheet() -> str:
    return "admod(Y, X) -> Additive model via marginal integration (Linton & Nielsen 1995)."
