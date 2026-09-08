# morie.fn -- function file (rootcoder007/morie)
"""RiskMetrics EWMA volatility."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vol_riskmetrics"]


def vol_riskmetrics(r, lam=0.94):
    r"""RiskMetrics exponentially weighted moving-average variance.

    .. math:: \sigma_t^2 = \lambda \sigma_{t-1}^2
              + (1 - \lambda) r_{t-1}^2,

    an IGARCH(1,1) with no intercept and coefficients summing to one.
    The default :math:`\lambda = 0.94` is the RiskMetrics daily
    ("trading") decay factor; the Technical Document also gives 0.97
    for monthly ("investing") horizons (p. 51, verified against the
    library PDF). Initialised at the sample variance of the first 20
    observations (or all, if fewer).

    Parameters
    ----------
    r : array-like, shape (n,)
        Return series.
    lam : float in (0, 1), default 0.94

    Returns
    -------
    RichResult
        keys: ``sigma2`` (n, conditional variance path), ``sigma``
        (its square root), ``forecast`` (one-step-ahead variance),
        ``lam``, ``n``, ``method``.

    References
    ----------
    J.P. Morgan/Reuters (1996). *RiskMetrics -- Technical Document*
    (4th ed.), p. 51 (decay factors 0.94 daily, 0.97 monthly; library
    PDF).
    """
    r = np.asarray(r, dtype=float).ravel()
    n = r.size
    if n < 2:
        raise ValueError("need at least 2 returns.")
    lam = float(lam)
    if not 0 < lam < 1:
        raise ValueError(f"lam must lie in (0, 1), got {lam}.")
    if not np.all(np.isfinite(r)):
        raise ValueError("r must be finite.")

    s2 = np.empty(n)
    s2[0] = r[: min(20, n)].var()
    if s2[0] <= 0:
        s2[0] = max(r[0] ** 2, 1e-12)
    for t in range(1, n):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2

    return RichResult(
        payload={
            "sigma2": s2,
            "sigma": np.sqrt(s2),
            "forecast": float(lam * s2[-1] + (1 - lam) * r[-1] ** 2),
            "lam": lam,
            "n": int(n),
            "method": "RiskMetrics EWMA variance (lambda = %.2f)" % lam,
        }
    )


def cheatsheet():
    return "volrm: s2_t = lam s2_{t-1} + (1-lam) r_{t-1}^2; lam=0.94 daily (RM 1996 p.51)"


# compact alias per ledger/NAMING.md
volriskmetrics = vol_riskmetrics
