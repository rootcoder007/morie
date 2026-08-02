# morie.fn -- function file (rootcoder007/morie)
"""EGARCH with a skewed generalized-error distribution."""

from . import _array_core as np
from scipy import optimize, special

from ._garch import garch_fit
from ._richresult import RichResult

__all__ = ["vol_nelson_skew_garch"]


def _skew_ged_loglik(z, nu, lam):
    """Skew-GED log-likelihood, Fernandez-Steel inverse-scale skewing.

    A single scale factor lam > 0 stretches one half of the density
    and compresses the other; lam = 1 is the symmetric GED, so the
    skew is testable by whether the fitted lam departs from 1.
    """
    if nu <= 0 or lam <= 0:
        return -np.inf
    a = np.sqrt(2 ** (-2 / nu) * special.gamma(1 / nu) / special.gamma(3 / nu))
    c = np.log(2) - np.log(lam + 1 / lam) + np.log(nu) - np.log(a) - (
        1 + 1 / nu
    ) * np.log(2) - special.gammaln(1 / nu)
    scaled = np.where(z < 0, z * lam, z / lam)
    return float(np.sum(c - 0.5 * np.abs(scaled / a) ** nu))


def vol_nelson_skew_garch(r):
    r"""EGARCH volatility with skew-GED standardised residuals.

    Fits Nelson's EGARCH (Tsay Sec. 3.8, p. 143) for the variance
    dynamics, then fits a skewed generalized error distribution to the
    standardised residuals in a second stage. Two distinct asymmetries
    are involved and the result reports both separately: the EGARCH
    ``gamma`` is asymmetry in how shocks move *volatility* (the
    leverage effect), while the skew ``lambda`` is asymmetry in the
    *innovation* distribution itself. A series can have one without
    the other.

    Parameters
    ----------
    r : array-like
        Return series.

    Returns
    -------
    RichResult
        keys: ``params`` (EGARCH), ``sigma2``, ``sigma``, ``nu``,
        ``lambda_skew``, ``skew_loglik``, ``symmetric_loglik``,
        ``skew_lr_stat`` (likelihood-ratio against lambda = 1),
        ``std_residuals``, ``persistence``, ``n``, ``method``.

    References
    ----------
    Nelson, D. B. (1991). Conditional heteroskedasticity in asset
    returns: a new approach. *Econometrica*, 59(2), 347-370.

    Fernandez, C. & Steel, M. F. J. (1998). On Bayesian modeling of fat
    tails and skewness. *Journal of the American Statistical
    Association*, 93(441), 359-371.
    """
    fit = garch_fit(r, "egarch")
    z = fit["std_residuals"]

    def neg(x):
        nu = 0.2 + 4.0 / (1 + np.exp(-np.clip(x[0], -30, 30)))
        lam = np.exp(np.clip(x[1], -3, 3))
        ll = _skew_ged_loglik(z, nu, lam)
        return 1e10 if not np.isfinite(ll) else -ll

    res = optimize.minimize(neg, [0.0, 0.0], method="Nelder-Mead",
                            options={"maxiter": 2000, "fatol": 1e-8})
    nu = 0.2 + 4.0 / (1 + np.exp(-np.clip(res.x[0], -30, 30)))
    lam = float(np.exp(np.clip(res.x[1], -3, 3)))
    sym = -optimize.minimize_scalar(
        lambda t: -_skew_ged_loglik(z, 0.2 + 4.0 / (1 + np.exp(-t)), 1.0),
        bounds=(-30, 30), method="bounded"
    ).fun

    out = dict(fit)
    out.update(
        {
            "nu": float(nu), "lambda_skew": lam, "skew_loglik": float(-res.fun),
            "symmetric_loglik": float(sym),
            "skew_lr_stat": float(2 * (-res.fun - sym)),
            "method": "EGARCH variance with skew-GED innovations (Nelson 1991)",
        }
    )
    return RichResult(payload=out)


def cheatsheet():
    return "volnsig: EGARCH gamma = volatility asymmetry; skew lambda = density asymmetry"
