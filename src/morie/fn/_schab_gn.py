# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gauss-Newton fitting of a semivariogram model.

Schabenberger & Gotway (2005), Sec. 4.5.1 and Sec. 4.5.3. The book names
the algorithm rather than leaving it open:

  "GEE estimates can thus be calculated as the ordinary (nonlinear) least
   squares estimates in the model T = 2 gamma(h, theta) + delta ... with a
   Gauss-Newton algorithm."                                    (after 4.43)

  "Composite likelihood estimates can thus be calculated by (nonlinear)
   weighted least squares in the model
   T = 2 gamma(h,theta) + delta, delta ~ (0, 8 gamma(h,theta)^2), with a
   Gauss-Newton algorithm."                                    (after 4.44)

and for the weighted case, Sec. 4.5.1: "an iterative re-weighting scheme
is employed since updates to theta should be followed by updates to
R(theta)". So the weights are recomputed at the top of every outer
iteration; they are not frozen at the starting values.

The estimating equation (4.42)/(4.43) is written in terms of
d gamma(h, theta) / d theta, so those derivatives are taken analytically
here. That matters for cross-language agreement: an analytic Jacobian has
no finite-difference step to choose, so R and Python execute the same
arithmetic rather than two differently-tuned approximations.

Everything here is internal.
"""

import numpy as np

from ._schab_vario import PRACTICAL_RANGE_C, correlogram, semivariogram

__all__ = []


def semivariogram_jacobian(h, nugget, sill, rng, model):
    """d gamma(h; c0, sigma0^2, a) / d(c0, sigma0^2, a), shape (len(h), 3).

    gamma(h) = c0 + sigma0^2 (1 - R(h; a)) for h > 0, so

        d gamma / d c0        = 1
        d gamma / d sigma0^2  = 1 - R(h; a)
        d gamma / d a         = -sigma0^2 dR/da

    and every row is zero at h = 0, where gamma(0) = 0 by definition
    whatever the nugget (Sec. 4.3.6).
    """
    h = np.asarray(h, dtype=float).ravel()
    c = PRACTICAL_RANGE_C
    r = correlogram(h, rng, model)
    if model == "exponential":                      # R = exp(-c h / a)
        dr_da = r * (c * h / rng**2)
    elif model == "gaussian":                       # R = exp(-c (h/a)^2)
        dr_da = r * (2.0 * c * h**2 / rng**3)
    elif model == "spherical":                      # R = 1 - 1.5u + 0.5u^3
        dr_da = np.zeros_like(h)
        inside = h <= rng
        hi = h[inside]
        dr_da[inside] = 1.5 * hi / rng**2 - 1.5 * hi**3 / rng**4
    else:
        raise ValueError(f"unknown model {model!r}")
    jac = np.column_stack([np.ones_like(h), 1.0 - r, -sill * dr_da])
    jac[h == 0.0, :] = 0.0
    return jac


def _weights(kind, ghat_model, counts):
    """W^-1 for the two criteria of Sec. 4.5.1.

    OLS is R = phi I, so the weights are 1. WLS uses Cressie's (1985)
    approximation (4.33), Var[gamma_hat(h_m)] = 2 gamma(h_m,theta)^2/|N(h_m)|,
    whose reciprocal is the weight in (4.34).
    """
    if kind == "ols":
        return np.ones_like(ghat_model)
    denom = 2.0 * ghat_model**2
    # A lag class where the fitted semivariogram is zero carries no
    # information under this criterion, and 1/0 would otherwise poison the
    # normal equations. The degenerate model itself is rejected in
    # `objective` below, so this only guards the arithmetic.
    return np.where(denom > 0, counts / np.where(denom > 0, denom, 1.0), 0.0)


def _project(theta):
    """Onto the parameter space of Sec. 4.3: variances >= 0, a range > 0.

    This is the constraint the model itself imposes, not a search box.
    """
    t = np.array(theta, dtype=float)
    t[0] = max(t[0], 0.0)
    t[1] = max(t[1], 0.0)
    t[2] = max(t[2], np.finfo(float).tiny)
    return t


def gauss_newton_semivariogram(lags, ghat, counts, start, model="exponential",
                               kind="wls", max_iter=200, tol=1e-12,
                               max_halvings=40):
    """Fit by Gauss-Newton with iterative re-weighting.

    Returns (theta, objective, converged, n_iter).
    """
    h = np.asarray(lags, dtype=float)
    g = np.asarray(ghat, dtype=float)
    n = np.asarray(counts, dtype=float)
    ok = np.isfinite(h) & np.isfinite(g) & (n > 0)
    h, g, n = h[ok], g[ok], n[ok]
    if h.size < 3:
        raise ValueError("need at least 3 usable lag classes to fit 3 parameters")

    def objective(theta):
        fitted = semivariogram(h, theta[0], theta[1], theta[2], model)
        if not np.all(np.isfinite(fitted)):
            return np.inf, fitted, np.zeros_like(fitted), np.zeros_like(fitted)
        if kind == "wls" and np.any(fitted <= 0.0):
            # gamma(h) = 0 at a positive lag means no nugget and no partial
            # sill: the model has collapsed and (4.34) is undefined there.
            return np.inf, fitted, np.zeros_like(fitted), np.zeros_like(fitted)
        w = _weights(kind, fitted, n)
        resid = g - fitted
        val = float(np.sum(w * resid**2))
        return (val if np.isfinite(val) else np.inf), fitted, w, resid

    theta = _project(start)
    obj, fitted, w, resid = objective(theta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        jac = semivariogram_jacobian(h, theta[0], theta[1], theta[2], model)
        # Normal equations of the weighted Gauss-Newton step:
        #   (J' W J) delta = J' W r
        jw = jac * w[:, None]
        lhs = jac.T @ jw
        rhs = jw.T @ resid
        if not (np.all(np.isfinite(lhs)) and np.all(np.isfinite(rhs))):
            break
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        # Step halving: the plain Gauss-Newton step is not guaranteed to
        # decrease a weighted objective whose weights also move, so shorten
        # it until it does. Halving is the standard safeguard, and the
        # sequence is fixed, so both language arms take the same steps.
        stepped = False
        for k in range(max_halvings):
            trial = _project(theta + delta / (2.0**k))
            trial_obj, t_fitted, t_w, t_resid = objective(trial)
            if np.isfinite(trial_obj) and trial_obj < obj:
                rel = (obj - trial_obj) / max(abs(obj), 1e-300)
                theta, obj = trial, trial_obj
                fitted, w, resid = t_fitted, t_w, t_resid
                stepped = True
                converged = rel < tol
                break
        if not stepped or converged:
            converged = True
            break
    return theta, obj, bool(converged), int(it)
