"""Comparing spatial covariance models: likelihood-ratio tests and AIC (eqs 6.57-6.60)."""

import math

from . import _array_core as np
from ._richresult import RichResult
from ._schab_st import _chi2_sf_1df
from ._schaben import _nelder_mead, gaussian_neg2loglik, reml_neg2loglik
from .spml import schabenberger_ml_variogram

__all__ = ["spatial_covariance_comparison"]


def spatial_covariance_comparison(coords, z, X=None, model="exponential", method="reml"):
    r"""Likelihood-ratio tests and AIC for nested spatial covariance models.

    Schabenberger & Gotway (2005, Sec. 6.2.3.1, eqs 6.57-6.60, Examples
    6.1 and 6.3-6.4) fit :math:`Z(s) = X(s)\beta + e(s)` under three nested
    covariance structures -- ``model`` with a nugget :math:`(c_0, \sigma^2,
    \alpha)`, without one :math:`(\sigma^2, \alpha)`, and independence
    :math:`(\sigma^2)` -- by ML or REML, and compare them.

    The likelihood-ratio statistic is the difference in :math:`\varphi`
    (minus twice the log likelihood; for REML :math:`\varphi_R`, valid
    because the mean structures match). For :math:`H_0: c_0 = 0` the
    nugget sits on the boundary of the parameter space, so the null
    distribution is a 50:50 mixture of a point mass at zero and
    :math:`\chi^2_1`, and the p-value is half the :math:`\chi^2_1` one
    (Self & Liang 1987). The independence test (Example 6.1) is reported
    against :math:`\chi^2_1` as in the book; under its null the range is
    not identified, so that reference distribution is approximate.

    AIC follows (6.59)-(6.60): :math:`\varphi + 2(k + q)` for ML and
    :math:`\varphi_R + 2q` for REML, with ``k`` mean and ``q`` covariance
    parameters.

    Parameters
    ----------
    coords : array-like, (n, 2)
    z : array-like, (n,)
    X : array-like, (n, k), optional
        Mean design, default an intercept.
    model : str
        ``"exponential"``, ``"gaussian"``, ``"spherical"`` or ``"linear"``.
    method : str
        ``"ml"`` or ``"reml"``.

    Returns
    -------
    RichResult
        ``fits`` (per model: ``neg2loglik``, ``q``, ``aic`` and the
        estimates), ``lrt_nugget``, ``p_nugget`` (boundary-corrected),
        ``p_nugget_naive``, ``lrt_spatial``, ``p_spatial``, ``k``.

    References
    ----------
    Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of maximum
    likelihood estimators and likelihood ratio tests under nonstandard
    conditions. JASA 82, 605-610. Schabenberger, O. & Gotway, C. A.
    (2005). Statistical Methods for Spatial Data Analysis. Chapman &
    Hall/CRC, eqs (6.57)-(6.60), pp. 343-345.
    """
    if method not in ("ml", "reml"):
        raise ValueError("`method` must be 'ml' or 'reml'")
    zz = np.asarray(z, dtype=float).ravel()
    n = int(zz.size)
    Xd = np.ones((n, 1)) if X is None else np.atleast_2d(np.asarray(X, dtype=float))
    if Xd.shape[0] != n:
        Xd = Xd.T
    k = int(Xd.shape[1])
    obj_fn = gaussian_neg2loglik if method == "ml" else reml_neg2loglik
    full = schabenberger_ml_variogram(coords, zz, model, method=method, X=Xd)

    v0 = float(np.var(zz))
    from ._schaben import pair_differences

    h, _ = pair_differences(coords, zz)
    hmax = float(np.max(h))

    def obj(p):
        val, _ = obj_fn(coords, zz, model, 0.0, math.exp(float(p[0])), math.exp(float(p[1])), Xd)
        return val if np.isfinite(val) else 1e12

    best, bestval = None, math.inf
    for frac in (0.1, 0.25, 0.5, 1.0):
        p, val = _nelder_mead(obj, np.log(np.array([v0, max(frac * hmax, 1e-6)])))
        if val < bestval:
            best, bestval = p, val
    psill, rng = math.exp(float(best[0])), math.exp(float(best[1]))
    phi_nonug, _ = obj_fn(coords, zz, model, 0.0, psill, rng, Xd)

    beta_ols = np.linalg.solve(Xd.T @ Xd, Xd.T @ zz)
    rss = float(np.sum((zz - Xd @ beta_ols) ** 2))
    s2 = rss / (n if method == "ml" else n - k)
    phi_ind, _ = obj_fn(coords, zz, model, s2, 0.0, 1.0, Xd)

    def aic(phi, q):
        return float(phi) + 2.0 * ((k + q) if method == "ml" else q)

    fits = {
        "nugget": {
            "neg2loglik": float(full["neg2loglik"]),
            "q": 3,
            "aic": aic(full["neg2loglik"], 3),
            "nugget": full["nugget"],
            "psill": full["psill"],
            "range": full["range"],
        },
        "no_nugget": {
            "neg2loglik": float(phi_nonug),
            "q": 2,
            "aic": aic(phi_nonug, 2),
            "nugget": 0.0,
            "psill": psill,
            "range": rng,
        },
        "independent": {"neg2loglik": float(phi_ind), "q": 1, "aic": aic(phi_ind, 1), "sigma2": s2},
    }
    lrt_n = max(float(phi_nonug) - float(full["neg2loglik"]), 0.0)
    lrt_s = max(float(phi_ind) - float(phi_nonug), 0.0)
    p_naive = float(_chi2_sf_1df(lrt_n))
    return RichResult(
        title="Nested spatial covariance models: LRT and AIC (eqs 6.57-6.60)",
        summary_lines=[("method", method.upper()), ("LRT nugget", lrt_n), ("LRT spatial", lrt_s)],
        payload={
            "fits": fits,
            "lrt_nugget": lrt_n,
            "p_nugget": 0.5 * p_naive,
            "p_nugget_naive": p_naive,
            "lrt_spatial": lrt_s,
            "p_spatial": float(_chi2_sf_1df(lrt_s)),
            "k": k,
            "method": method,
            "model": model,
        },
    )


def cheatsheet():
    return "spcmp: LRT (boundary 50:50 for the nugget) and AIC (6.59/6.60) for nested spatial covariance models"
