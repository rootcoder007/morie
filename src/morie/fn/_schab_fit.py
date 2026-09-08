# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared machinery for fitting a semivariogram model to data.

Schabenberger & Gotway (2005), *Statistical Methods for Spatial Data
Analysis*, Sec. 4.4 (least-squares fitting) and Sec. 4.5 (likelihood).

References
----------
Schabenberger, O. & Gotway, C. A. (2005) *Statistical Methods for
Spatial Data Analysis*, Texts in Statistical Science, Chapman &
Hall/CRC, Boca Raton, ISBN 1-58488-322-7.
Sec. 4.4 (least-squares fitting) and Sec. 4.5 (likelihood).

Harville, D. A. (1974) "Bayesian inference for variance components
using only error contrasts", *Biometrika* 61, 383-385 -- the error
contrasts that REML is built on.

Everything here is internal; the public entry points are `spols`, `spwls`
and `spreml`.
"""

from . import _array_core as np

from ._schab_gn import gauss_newton_semivariogram
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
    """Fit by Gauss-Newton, the algorithm Sec. 4.5 names for this problem.

    The text does not leave the numerical method open: the GEE/OLS estimates
    (after 4.43) and the composite-likelihood/WLS estimates (after 4.44) are
    both to be "calculated ... with a Gauss-Newton algorithm", and Sec. 4.5.1
    adds that the weights must be refreshed as theta moves. That is what
    `gauss_newton_semivariogram` does, with the derivatives of (4.42) taken
    analytically.

    Returns (nugget, partial_sill, range, objective_value, converged).
    """
    if kind not in ("ols", "wls"):
        raise ValueError("`kind` must be 'ols' or 'wls'")
    f, ok = _objective(kind, lags, ghat, counts, model)
    if ok.sum() < 3:
        raise ValueError("need at least 3 usable lag classes to fit 3 parameters")
    start, _ = _start_and_bounds(np.asarray(lags)[ok], np.asarray(ghat)[ok])
    theta, obj, converged, _ = gauss_newton_semivariogram(
        lags, ghat, counts, start, model=model, kind=kind)
    return (float(theta[0]), float(theta[1]), float(theta[2]),
            float(obj), bool(converged))


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
