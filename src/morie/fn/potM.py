"""Peaks-over-threshold GPD analysis (Pickands 1975; Davison & Smith 1990)."""

import math

from ._evt_core import gpd_mle
from ._richresult import RichResult

__all__ = ["potM", "peaks_over_threshold"]


def potM(y, u, return_periods=(10.0, 100.0)):
    """
    Peaks-over-threshold analysis: GPD fit to threshold exceedances.

    The exceedances of a high threshold u are asymptotically
    generalized Pareto (Pickands 1975); Davison & Smith (1990)
    developed the resulting regression/inference framework.  This
    front-end extracts the exceedances y_i - u for y_i > u, fits the
    GPD by maximum likelihood via the shelf-verified core
    (morie.fn._evt_core.gpd_mle, Coles Sec. 4.3.2), estimates the
    exceedance rate zeta_u = k/n, and computes m-observation return
    levels (Coles 2001, Eq. 4.13):

        x_m = u + (sigma/xi) [ (m zeta_u)^xi - 1 ]        (xi != 0)
        x_m = u + sigma log(m zeta_u)                     (xi == 0).

    Sources
    -------
    Pickands, J. (1975). Statistical inference using extreme order
    statistics. *Annals of Statistics*, 3, 119-131.
    Davison, A. C. & Smith, R. L. (1990). Models for exceedances
    over high thresholds. *JRSS-B*, 52, 393-442.
    Coles, S. (2001). *An Introduction to Statistical Modeling of
    Extreme Values*, Springer, Sec. 4.2-4.3 (GPD threshold model,
    Eq. 4.13 return levels; the shelf's primary source).

    Parameters
    ----------
    y : sequence of float
        Raw observations.
    u : float
        Threshold; exceedances are y_i - u for y_i > u.
    return_periods : sequence of float
        Return periods m (in numbers of observations) for x_m.

    Returns
    -------
    RichResult
        Keys: sigma, xi, loglik, cov, n_exceedances, rate,
        return_levels ({m: x_m}), threshold.
    """
    yv = [float(v) for v in y]
    u = float(u)
    n = len(yv)
    exc = [v - u for v in yv if v > u]
    k = len(exc)
    if k < 2:
        raise ValueError("need at least two exceedances above u")
    fit = gpd_mle(exc)
    sigma = fit["sigma"]
    xi = fit["xi"]
    rate = k / float(n)
    rl = {}
    for m in return_periods:
        m = float(m)
        if m * rate <= 1.0:
            rl[m] = float("nan")
            continue
        if abs(xi) < 1e-9:
            rl[m] = u + sigma * math.log(m * rate)
        else:
            rl[m] = u + sigma / xi * ((m * rate) ** xi - 1.0)
    return RichResult(payload={
        "sigma": sigma, "xi": xi,
        "loglik": fit["loglik"], "cov": fit["cov"],
        "n_exceedances": k, "n": n, "rate": rate,
        "return_levels": rl, "threshold": u,
        "converged": fit["converged"],
        "method": "POT/GPD (Davison-Smith 1990; Coles Eq. 4.13)",
    })


# long descriptive alias (stub-era name)
peaks_over_threshold = potM


def cheatsheet():
    return "potM: GPD MLE on y-u | y>u; x_m = u + sigma/xi ((m zeta)^xi - 1)"
