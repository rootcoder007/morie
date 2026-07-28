# morie.fn -- function file (rootcoder007/morie)
"""Covariance-parameter estimation for kriging (LS, ML, REML)."""

import numpy as np

from ._richresult import RichResult
# imported under an alias: the public argument is also called
# variogram_model, and letting the string shadow the function makes the
# covariance assembly fail only on the spatially-varying-mean path
from ._schaben import MODELS, fit_variogram_wls, matheron
from ._schaben import variogram_model as _vgm
from ._did import add_intercept, ols_fit

__all__ = ["schabenberger_cov_param_estimation_kriging"]

METHODS = ("wls", "ols", "ml", "reml", "cl")


def schabenberger_cov_param_estimation_kriging(coords, z,
                                               variogram_model="exponential",
                                               method="reml", X=None,
                                               max_iter=25, tol=1e-8):
    r"""Covariance parameters for kriging, Schabenberger section 5.5.

    With a constant mean the section 4.5 machinery applies directly.
    With a spatially varying mean it does not, and the reason is
    equation (5.35):

    .. math::
       E[(Z(s_i)-Z(s_j))^2] = 2\gamma(s_i-s_j)
         + \Big\{\sum_k \beta_k (x_k(s_i)-x_k(s_j))\Big\}^{2},

    so the empirical semivariogram of the RAW data no longer estimates
    the semivariogram at all -- the trend leaks into it, typically
    making it rise quadratically with distance.

    Escaping that needs the semivariogram of the errors, which needs
    :math:`\beta`, which needs the covariance parameters. Schabenberger
    and Pierce call this the "cat and mouse game of universal kriging".
    The way out is the IRWGLS algorithm of p. 257:

    1. start from an estimate of :math:`\beta` (ordinary least squares);
    2. form residuals :math:`r = Z - X\hat\beta`;
    3. estimate and model the semivariogram OF THE RESIDUALS;
    4. re-estimate :math:`\beta` by estimated generalised least
       squares, equation (5.38);
    5. repeat 2-4 until the change is small.

    Two honest caveats the book insists on. The residual-based
    semivariogram is itself biased -- OLS residuals share only a zero
    mean with the true errors, and the bias grows with the lag, which
    is why weighted fitting (which down-weights long lags) is the
    default here. And the iteration is not an optimiser: there is no
    guarantee it converges to anything, so when it stops "think of it
    as lack of progress, rather than convergence". ``converged``
    reports the stopping test, not a claim of optimality.

    Parameters
    ----------
    coords : array-like, shape (n, d)
    z : array-like, shape (n,)
    variogram_model : {'exponential', 'spherical', 'gaussian', 'linear'}
    method : {'reml', 'ml', 'wls', 'ols', 'cl'}
        How the covariance parameters are estimated at each pass.
    X : array-like, optional
        Mean design matrix. Constant mean by default, in which case a
        single pass is all that is needed.
    max_iter, tol :
        IRWGLS controls.

    Returns
    -------
    RichResult
        ``parameters``, ``nugget``, ``psill``, ``range``, ``sill``,
        ``beta``, ``beta_se``, ``iterations``, ``converged``,
        ``trend_bias_warning``.

    References
    ----------
    Schabenberger and Gotway (2005), section 5.5 and 5.5.1-5.5.3,
    equations (5.35)-(5.40), pp. 254-263. Cressie (1993, p. 166).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(3)
    >>> co = rng.uniform(0, 10, size=(80, 2))
    >>> z = 2 + 0.5 * co[:, 0] + rng.normal(scale=0.5, size=80)
    >>> out = schabenberger_cov_param_estimation_kriging(
    ...     co, z, method="wls", X=co[:, :1])
    >>> bool(out["beta"].size == 2)
    True
    """
    model = variogram_model
    if model not in MODELS:
        raise ValueError("model must be one of %s, got %r." % (MODELS, model))
    if method not in METHODS:
        raise ValueError("method must be one of %s, got %r." % (METHODS, method))
    zz = np.asarray(z, dtype=float).ravel()
    n = zz.size
    Xd = np.ones((n, 1)) if X is None else add_intercept(
        np.atleast_2d(np.asarray(X, dtype=float))
    )
    if Xd.shape[0] != n:
        Xd = Xd.T
    varying = Xd.shape[1] > 1

    def _cov_params(resid):
        if method in ("ml", "reml"):
            from .spml import schabenberger_ml_variogram
            f = schabenberger_ml_variogram(coords, resid, model, method)
            return f["nugget"], f["psill"], f["range"]
        if method == "cl":
            from ._schaben import composite_likelihood_fit
            f = composite_likelihood_fit(coords, resid, model)
            return f["nugget"], f["psill"], f["range"]
        lag, gam, npair, _ = matheron(coords, resid)
        f = fit_variogram_wls(lag, gam, npair, model,
                              "ols" if method == "ols" else "cressie")
        return f["nugget"], f["psill"], f["range"]

    beta = ols_fit(Xd, zz)
    theta = None
    converged, it = False, 0
    for it in range(1, int(max_iter) + 1):
        r = zz - Xd @ beta
        new = _cov_params(r)
        P = np.atleast_2d(np.asarray(coords, dtype=float))
        if P.shape[0] != n:
            P = P.T
        D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
        sill = new[0] + new[1]
        C = sill - _vgm(D, model, *new)
        C = C + np.eye(n) * 1e-10 * max(sill, 1e-12)
        try:
            CiX = np.linalg.solve(C, Xd)
            XtCiX = Xd.T @ CiX
            beta_new = np.linalg.solve(XtCiX, CiX.T @ zz)
        except np.linalg.LinAlgError:
            beta_new, XtCiX = beta, Xd.T @ Xd
        moved = (np.max(np.abs(beta_new - beta))
                 + (0.0 if theta is None
                    else float(np.max(np.abs(np.array(new) - np.array(theta))))))
        beta, theta = beta_new, new
        if not varying or moved < tol:
            converged = True
            break

    P = np.atleast_2d(np.asarray(coords, dtype=float))
    if P.shape[0] != n:
        P = P.T
    D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    sill = theta[0] + theta[1]
    C = sill - _vgm(D, model, *theta)
    C = C + np.eye(n) * 1e-10 * max(sill, 1e-12)
    CiX = np.linalg.solve(C, Xd)
    cov_beta = np.linalg.inv(Xd.T @ CiX)          # equation (5.40)
    return RichResult(
        payload={
            "estimate": np.array([theta[0], theta[1], theta[2]]),
            "parameters": {"nugget": float(theta[0]), "psill": float(theta[1]),
                           "range": float(theta[2])},
            "nugget": float(theta[0]),
            "psill": float(theta[1]),
            "range": float(theta[2]),
            "sill": float(sill),
            "beta": np.atleast_1d(beta),
            "beta_cov": cov_beta,
            "beta_se": np.sqrt(np.diag(cov_beta)),
            "beta_se_note": (
                "equation (5.40); it treats the covariance parameters as "
                "known, so it understates the true uncertainty in beta"
            ),
            "method_used": method,
            "model": model,
            "iterations": it,
            "converged": converged,
            "convergence_note": (
                "the IRWGLS loop is not an optimiser and has no extremum to "
                "find; stopping means lack of progress, not optimality"
            ),
            "spatially_varying_mean": varying,
            "trend_bias_warning": (
                "the semivariogram is estimated from residuals, which are "
                "rank-deficient and heteroscedastic; the bias grows with the "
                "lag, which is why long lags are down-weighted"
                if varying else None
            ),
            "n": n,
            "method": "Covariance-parameter estimation for kriging (%s)"
                      % method.upper(),
        }
    )


def cheatsheet():
    return (
        "spkce: covariance parameters for kriging by WLS/ML/REML/CL inside "
        "the IRWGLS loop, with the trend-leakage bias of (5.35) stated"
    )
