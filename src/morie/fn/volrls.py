# morie.fn -- function file (rootcoder007/morie)
"""Recursive (exponentially forgetting) volatility update."""

import numpy as np

from ._richresult import RichResult

__all__ = ["vol_recursive_least_sq"]


def vol_recursive_least_sq(r, lam=0.97):
    r"""Recursive update on squared returns with forgetting factor lam.

    .. math:: \hat\sigma_t^2 = \lambda \hat\sigma_{t-1}^2
              + (1 - \lambda)\, r_t^2,

    the RLS/exponential-forgetting solution of the constant-variance
    recursive regression -- identical in form to RiskMetrics EWMA but
    updated with the *current* squared return (a nowcast) where
    RiskMetrics uses the lagged one (a forecast). The docstring says
    which convention this is so the two are never confused.

    Parameters
    ----------
    r : array-like, shape (n,)
        Return series.
    lam : float in (0, 1), default 0.97

    Returns
    -------
    RichResult
        keys: ``sigma2`` (n,), ``sigma``, ``lam``,
        ``effective_window`` (1/(1-lam)), ``n``, ``method``.

    References
    ----------
    Ljung, L. & Soederstroem, T. (1983). *Theory and Practice of
    Recursive Identification*. MIT Press. (exponential forgetting)

    J.P. Morgan/Reuters (1996). *RiskMetrics -- Technical Document*
    (4th ed.), p. 51. (the forecasting twin of this recursion)
    """
    r = np.asarray(r, dtype=float).ravel()
    n = r.size
    if n < 2:
        raise ValueError("need at least 2 returns.")
    lam = float(lam)
    if not 0 < lam < 1:
        raise ValueError(f"lam must lie in (0, 1), got {lam}.")

    s2 = np.empty(n)
    s2[0] = r[: min(20, n)].var()
    if s2[0] <= 0:
        s2[0] = max(r[0] ** 2, 1e-12)
    for t in range(1, n):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t] ** 2

    return RichResult(
        payload={
            "sigma2": s2,
            "sigma": np.sqrt(s2),
            "lam": lam,
            "effective_window": float(1.0 / (1.0 - lam)),
            "n": int(n),
            "method": f"Recursive squared-return update (forgetting factor {lam})",
        }
    )


def cheatsheet():
    return "volrls: s2_t = lam s2_{t-1} + (1-lam) r_t^2 (nowcast twin of RiskMetrics)"
