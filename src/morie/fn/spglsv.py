"""Generalized least squares fit of a semivariogram model, Cressie (1985)."""

import math

from ._richresult import RichResult
from ._schab_glsvg import chol_quad, cholesky, lag_pairs, matheron, matheron_covariance, model_gamma
from ._schab_npvg import nelder_mead

__all__ = ["gls_semivariogram_fit"]


def gls_semivariogram_fit(coords, z, breaks, model="exponential", nugget=True, start=None, max_reweight=100, tol=1e-6):
    r"""Fit a semivariogram model to the Matheron estimator by generalized least squares.

    Schabenberger & Gotway (2005, Sec. 4.5.1, eqs 4.30-4.32): the empirical
    semivariogram :math:`\hat\gamma(h) = \gamma(h, \theta) + e(h)` has error
    covariance :math:`R(\theta)`, and the GLS criterion is

    .. math::

        (\hat\gamma(h) - \gamma(h, \theta))' R(\theta)^{-1}
        (\hat\gamma(h) - \gamma(h, \theta)).

    For Gaussian data :math:`\mathrm{Cov}[T_{ij}^2, T_{kl}^2] = 2\{\gamma(s_i
    - s_l) + \gamma(s_j - s_k) - \gamma(s_i - s_k) - \gamma(s_j - s_l)\}^2`
    with :math:`T_{ij} = Z(s_i) - Z(s_j)` (Cressie 1993, eq 2.6.10; the
    printed (4.32) has :math:`\gamma(h_{ij})` for the first term), which
    gives :math:`R(\theta)` exactly. Because R depends on :math:`\theta`,
    the fit iterates: fix :math:`R(\hat\theta)`, minimise over
    :math:`\theta` by Nelder-Mead on the log scale, update R, until the
    estimates settle. Cost grows with the square of the number of pairs.

    Parameters
    ----------
    coords : sequence of (x, y)
    z : sequence of float
    breaks : sequence of float
        Lag-class boundaries; class m is (breaks[m], breaks[m+1]].
    model : str
        ``"exponential"``, ``"gaussian"``, ``"spherical"``, ``"wave"``,
        ``"tent"`` or ``"circular"``.
    nugget : bool
        Estimate a nugget (else it is zero).
    start : tuple, optional
        Starting ``(nugget, sill, range)``; by default a quarter of the
        first-class value, the largest value, and half the largest lag.
    max_reweight : int
    tol : float
        Change in the estimates, relative to each estimate or to
        ``1e-8`` times the largest one, that ends the re-weighting.

    Returns
    -------
    RichResult
        ``nugget``, ``sill`` (partial), ``range``, ``lags``, ``gamma_hat``,
        ``fitted``, ``npairs``, ``R`` (at the estimate), ``criterion``,
        ``iterations``.

    References
    ----------
    Cressie, N. (1985). Fitting variogram models by weighted least squares.
    Mathematical Geology 17, 563-586. Cressie, N. (1993). Statistics for
    Spatial Data, revised ed. Wiley, eq (2.6.10). Schabenberger, O. &
    Gotway, C. A. (2005). Statistical Methods for Spatial Data Analysis.
    Chapman & Hall/CRC, eqs (4.30)-(4.32), pp. 164-165.
    """
    z = [float(v) for v in z]
    pts, classes, lags = lag_pairs(coords, breaks)
    if len(z) != len(pts):
        raise ValueError("`coords` and `z` must have the same length")
    k = len(classes)
    npar = 3 if nugget else 2
    if k <= npar:
        raise ValueError("need more non-empty lag classes than parameters")
    ghat = matheron(z, classes)
    if start is None:
        start = (0.25 * ghat[0], max(ghat), 0.5 * max(lags))
    theta = [float(v) for v in start] if nugget else [0.0] + [float(v) for v in start[1:]]
    if any(v <= 0 for v in theta[1:]) or theta[0] < 0:
        raise ValueError("`start` values must be positive")

    def unpack(p):
        return [math.exp(p[0]), math.exp(p[1]), math.exp(p[2])] if nugget else [0.0, math.exp(p[0]), math.exp(p[1])]

    crit = None
    it = 0
    for step in range(1, int(max_reweight) + 1):
        it = step
        L = cholesky(matheron_covariance(pts, classes, theta[0], theta[1], theta[2], model))

        def q(p, L=L):
            th = unpack(p)
            fit = model_gamma(lags, th[0], th[1], th[2], model)
            return chol_quad(L, [a - b for a, b in zip(ghat, fit)])

        p0 = [math.log(max(v, 1e-12)) for v in (theta if nugget else theta[1:])]
        best, crit = nelder_mead(q, p0)
        new = unpack(best)
        scale = max(abs(v) for v in theta)
        change = max(abs(a - b) / max(abs(b), 1e-8 * scale) for a, b in zip(new, theta))
        theta = new
        if change < tol:
            break
    R = matheron_covariance(pts, classes, theta[0], theta[1], theta[2], model)
    fitted = model_gamma(lags, theta[0], theta[1], theta[2], model)
    return RichResult(
        title="Semivariogram fit by generalized least squares (Cressie 1985)",
        summary_lines=[("model", model), ("nugget", theta[0]), ("sill", theta[1]), ("range", theta[2])],
        payload={
            "nugget": theta[0],
            "sill": theta[1],
            "range": theta[2],
            "lags": lags,
            "gamma_hat": ghat,
            "fitted": fitted,
            "npairs": [len(c) for c in classes],
            "R": R,
            "criterion": crit,
            "iterations": it,
            "model": model,
        },
    )


def cheatsheet():
    return "spglsv: GLS semivariogram fit with Cressie's full covariance of the Matheron estimator"
