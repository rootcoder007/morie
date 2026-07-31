# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared machinery for fitting a semivariogram model to data.

Schabenberger & Gotway (2005), *Statistical Methods for Spatial Data
Analysis*, Sec. 4.4 (least-squares fitting) and Sec. 4.5 (likelihood).

Everything here is internal; the public entry points are `spols`, `spwls`
and `spreml`.
"""

import numpy as np
from scipy.optimize import minimize

from ._schab_vario import correlogram, semivariogram

__all__ = []


def as_empirical_variogram(ev):
    """Accept the mapping the empirical estimator returns, or a plain array.

    Counts default to 1 so an unweighted table still fits; that makes the WLS
    weights degenerate to 1/(2 gamma^2), which is the right limiting form
    rather than a silent failure.
    """
    if hasattr(ev, "keys"):
        lags = np.asarray(ev["lags"], dtype=float).ravel()
        gamma = np.asarray(ev["gamma"], dtype=float).ravel()
        counts = (np.asarray(ev["counts"], dtype=float).ravel()
                  if "counts" in ev else np.ones_like(lags))
        return lags, gamma, counts
    # `empirical_semivariogram` returns (lag, gamma, count) as a tuple, so
    # accept that shape directly rather than letting it fall through to the
    # column-array branch, where it would transpose into nonsense.
    if isinstance(ev, (tuple, list)) and len(ev) in (2, 3) \
            and all(np.ndim(part) == 1 for part in ev):
        lags = np.asarray(ev[0], dtype=float)
        gamma = np.asarray(ev[1], dtype=float)
        counts = (np.asarray(ev[2], dtype=float) if len(ev) == 3
                  else np.ones_like(lags))
        return lags, gamma, counts
    arr = np.atleast_2d(np.asarray(ev, dtype=float))
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            "`empirical_variogram` needs at least lag and gamma columns")
    counts = arr[:, 2] if arr.shape[1] > 2 else np.ones(arr.shape[0])
    return arr[:, 0], arr[:, 1], counts


def _start_and_bounds(lags, ghat):
    """A scale-free starting point and the valid parameter space.

    The bounds ARE the parameter space of Sec. 4.3: a nugget and a partial
    sill are variances so they are non-negative, and a range is a distance so
    it is strictly positive. Nothing narrower -- a fit that cannot reach
    sill = 0 cannot report "no spatial structure", and one that cannot reach
    nugget = 0 cannot report a continuous field.
    """
    g = np.asarray(ghat, dtype=float)
    h = np.asarray(lags, dtype=float)
    finite = np.isfinite(g)
    if not finite.any():
        raise ValueError("empirical semivariogram is entirely non-finite")
    gmax = float(np.nanmax(g[finite]))
    hmax = float(np.nanmax(h[finite]))
    start = np.array([0.1 * gmax, 0.9 * gmax, 0.5 * hmax])
    # (nugget, partial sill, range); the upper limits are generous rather
    # than informative -- they exist so the optimiser stays finite.
    bounds = [(0.0, 10.0 * gmax + 1.0),
              (0.0, 10.0 * gmax + 1.0),
              (1e-8 * hmax + 1e-12, 10.0 * hmax)]
    return start, bounds


def _residuals(kind, lags, ghat, counts, model):
    """Residual vector whose sum of squares is the objective.

    The text calls semivariogram fitting a nonlinear least squares problem, so
    keeping the residual form lets a trust-region solver use the structure
    instead of hammering a scalar objective with numerical gradients. That is
    not cosmetic: L-BFGS-B on the scalar form stops after one iteration here,
    reporting the starting values as the fit.
    """
    h = np.asarray(lags, dtype=float)
    g = np.asarray(ghat, dtype=float)
    n = np.asarray(counts, dtype=float)
    ok = np.isfinite(g) & np.isfinite(h) & (n > 0)
    h, g, n = h[ok], g[ok], n[ok]

    def r(theta):
        nugget, sill, rng = theta
        fitted = semivariogram(h, max(nugget, 0.0), max(sill, 0.0),
                               max(rng, 1e-12), model)
        resid = g - fitted
        if kind == "ols":
            return resid
        # eq (4.34) as a sum of squares: each term is
        # |N(h_m)| / (2 gamma^2) * resid^2, so the residual carries the
        # square root of that weight. gamma -> 0 only when the model is
        # degenerate (no nugget and no sill); clip so the solver sees a
        # large residual there rather than a non-finite one.
        denom = np.sqrt(2.0) * np.maximum(fitted, 1e-12)
        return np.sqrt(n) * resid / denom

    return r, ok


def _objective(kind, lags, ghat, counts, model):
    h = np.asarray(lags, dtype=float)
    g = np.asarray(ghat, dtype=float)
    n = np.asarray(counts, dtype=float)
    ok = np.isfinite(g) & np.isfinite(h) & (n > 0)
    h, g, n = h[ok], g[ok], n[ok]

    def f(theta):
        nugget, sill, rng = theta
        if nugget < 0 or sill < 0 or rng <= 0:
            return np.inf
        fitted = semivariogram(h, nugget, sill, rng, model)
        resid = g - fitted
        if kind == "ols":
            # R = phi * I: the OLS simplification named in the text after
            # eq (4.34), which ignores both the correlation and the unequal
            # dispersion among the gamma-hat(h_m).
            return float(resid @ resid)
        # eq (4.34): sum_m |N(h_m)| / (2 gamma(h_m,theta)^2) * resid_m^2.
        # The weights are functions of theta, which is what makes this a
        # re-weighted rather than a plain weighted fit.
        denom = 2.0 * fitted**2
        good = denom > 0
        if not good.any():
            return np.inf
        return float(np.sum(n[good] * resid[good] ** 2 / denom[good]))

    return f, ok


def fit_semivariogram(lags, ghat, counts, model="exponential", kind="wls"):
    """Minimise (4.34) for `kind="wls"` or its OLS simplification.

    Returns (nugget, partial_sill, range, objective_value, converged).
    """
    if kind not in ("ols", "wls"):
        raise ValueError("`kind` must be 'ols' or 'wls'")
    f, ok = _objective(kind, lags, ghat, counts, model)
    if ok.sum() < 3:
        raise ValueError("need at least 3 usable lag classes to fit 3 parameters")
    start, bounds = _start_and_bounds(np.asarray(lags)[ok], np.asarray(ghat)[ok])
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def bounded(theta):
        """Bounds enforced inside the objective, not by the solver.

        R's optim() ignores bounds under Nelder-Mead, so pushing the box into
        the function is what lets both language arms run the identical search
        rather than two solvers that merely agree in intent.
        """
        t = np.asarray(theta, dtype=float)
        if np.any(t < lo) or np.any(t > hi) or not np.all(np.isfinite(t)):
            return np.inf
        return f(t)

    # Several starts: these objectives have a flat ridge along (nugget + sill)
    # and a simplex launched onto it stalls. The starts span the nugget
    # fraction, which is the direction the ridge runs in.
    best_x, best_f = np.asarray(start, dtype=float), bounded(start)
    for frac in (0.0, 0.1, 0.3, 0.6):
        for rscale in (0.25, 0.5, 1.0):
            x0 = np.clip(np.array([frac * start[1], start[1],
                                   rscale * 2.0 * start[2]]), lo, hi)
            res = minimize(bounded, x0, method="Nelder-Mead",
                           options={"maxiter": 4000, "maxfev": 4000,
                                    "xatol": 1e-10, "fatol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_f:
                best_x, best_f = np.asarray(res.x, dtype=float), float(res.fun)
    nugget, sill, rng = (float(v) for v in best_x)
    converged = bool(best_f < bounded(start)) or bool(np.allclose(best_x, start))
    return nugget, sill, rng, float(best_f), converged


def covariance_matrix(coords, nugget, sill, rng, model):
    """Sigma(theta) for the model of Sec. 4.3.

    C(0) = c0 + sigma0^2 and C(h) = sigma0^2 R(h) for h > 0, so the nugget
    enters only on the diagonal -- it is a discontinuity at the origin, not a
    term added everywhere.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    sigma = sill * correlogram(d, rng, model)
    sigma[np.diag_indices_from(sigma)] = nugget + sill
    return sigma


def error_contrasts(X):
    """A matrix K of error contrasts: full row rank, K X = 0.

    Sec. 4.5.2 builds K explicitly for the intercept-only case and notes,
    citing Harville (1974), that the choice of K does not matter for
    estimation. Taking the orthogonal complement of the column space of X
    gives one such K for any linear mean structure.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.ndim != 2:
        raise ValueError("`X` must be a 2-D design matrix")
    n, p = X.shape
    u, s, _ = np.linalg.svd(X, full_matrices=True)
    rank = int((s > max(n, p) * np.finfo(float).eps * (s[0] if s.size else 1.0)).sum())
    if rank >= n:
        raise ValueError("design matrix leaves no error contrasts")
    return u[:, rank:].T
